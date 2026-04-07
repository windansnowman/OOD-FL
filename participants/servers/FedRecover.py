from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer
import random
import logging
import time
import copy
import models.resnet
import models.vgg
from utils.utils import add_trigger
import numpy as np
import torch.optim as optim
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("logger")


class FedRecover(AbstractServer):

    def __init__(self, params, current_time, train_dataset, blend_pattern,
                 edge_case_train, edge_case_test):
        super(FedRecover, self).__init__(params, current_time)
        self.train_dataset = train_dataset
        self.blend_pattern = blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

        self.no_detected_malicious = 0
        self.no_undetected_malicious = 0
        self.no_detected_benign = 0
        self.no_misclassified_benign = 0
        self.no_processed_malicious_clients = 0
        self.no_processed_benign_clients = 0
        self._create_check_model()

        # FedRecover related
        self.malicious_clients = [0, 1]  # List of malicious client IDs
        self.historical_updates = {i: [] for i in
                                   self.malicious_clients}  # Store historical updates for malicious clients only
        self.recovering = False  # Flag to indicate if we are in the recovering phase
        self.recovering_start_round = params["poisoned_start_round"]
        self.recovering_end_round = params["poisoned_end_round"]
        # self.known_malicious_client = 0  # Assuming client 0 is the known malicious client
        self.benign_clients = [i for i in range(params["no_of_total_participants"]) if
                               i not in self.malicious_clients]  # list of benign clients
        self.pre_recover_rounds = params["poisoned_start_round"]  # Collect pre-attack rounds
        self.server_model_history = []  # save server model
        self.local_model_history = {i: [] for i in range(params["no_of_total_participants"])}  # save local model
        self.global_model_at_start = None  # save global model at the start
        self.local_model_at_start = {i: None for i in
                                     range(params["no_of_total_participants"])}  # save local model at the start
        self.local_update_history = {i: [] for i in self.malicious_clients}  # save local update history, malicious only
        self.estimated_updates = {}
        self.global_model_history = {}

    def _create_check_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10,
                                                                                dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=100,
                                                                                dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10,
                                                                                dataset="EMNIST")
        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)

        self.check_model = check_model.cuda()
        return True

    def _select_clients(self, round):
        r"""
        randomly select participating clients for each round
        """
        adversary_list = self.malicious_clients \
            if round in self.poisoned_rounds else []

        selected_clients = self.malicious_clients + random.sample(
            [i for i in range(self.params["no_of_total_participants"]) if i not in self.malicious_clients],
            self.params["no_of_participants_per_round"] - len(self.malicious_clients))
        return selected_clients, adversary_list

    def aggregation(self, weight_accumulator, aggregated_model_id, round):
        r"""
        aggregate all the updates model to generate a new global model
        """
        no_of_participants_this_round = len(aggregated_model_id)
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * \
                               (self.params["eta"] / no_of_participants_this_round)

            data = data.float()
            data.add_(update_per_layer)
        self.global_model_history = self.global_model.state_dict()
        return True

    def _norm_clip(self, local_client, round, model_id):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_client.local_model.named_parameters():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        scale = max(1.0, float(torch.abs(l2_norm / self.params["norm_bound"])))
        logger.info(f"round:{round}, local model {model_id} | l2_norm: {l2_norm}, scale: {scale}")

        if self.params["norm_clip"]:
            for name, data in local_client.local_model.named_parameters():
                new_value = self.global_model.state_dict()[name] + (
                        local_client.local_model.state_dict()[name] - self.global_model.state_dict()[name]) / scale
                local_client.local_model.state_dict()[name].copy_(new_value)

        return True

    def local_data_distrib(self, train_data):
        ### Initialization
        distrib_dict = dict()
        no_class = 100 if self.params["dataset"].upper() == "CIFAR100" else 10
        for label in range(no_class):
            distrib_dict[label] = 0

        ### count the class distribution
        for batch_id, batch in enumerate(train_data):
            _, targets = batch
            for target in targets:
                distrib_dict[int(target.item())] += 1
        ### count sum
        sum_no = 0
        for key, value in distrib_dict.items():
            sum_no += value

        ### count percentage
        percentage_dict = dict()
        for key, value in distrib_dict.items():
            percentage_dict[key] = round(value / sum_no, 2)

        return distrib_dict, percentage_dict, sum_no

    def _cos_sim(self, client, target_params_variables):
        model_list = []
        poison_dir_list = []
        for key, value in client.local_model.named_parameters():
            model_list.append(value.view(-1))
            poison_dir_list.append(target_params_variables[key].view(-1))

        model_tensor = torch.cat(model_list).cuda()
        poison_dir_tensor = torch.cat(poison_dir_list).cuda()
        cs = F.cosine_similarity(model_tensor, poison_dir_tensor, dim=0)
        return cs

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader,
                         test_dataloader, poison_train_dataloader, former_model):

        r"""
        Server broadcasts the global model to all participants.
        Every participants train its our local model and upload the weight difference to the server.
        The server then aggregate the changes in the weight_accumulator and return it.
        """
        ### Log info
        logger.info(f"Training on global round {round} begins")

        ### Count adversaries in one global round
        current_no_of_adversaries = 0
        selected_clients, adversary_list = self._select_clients(round)
        for client_id in selected_clients:
            if client_id in self.malicious_clients:
                current_no_of_adversaries += 1
        logger.info(f"There are {current_no_of_adversaries} adversaries in the training for round {round}")

        ### Initialize the accumulator for all participants
        weight_accumulator = dict()
        for name, data in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        ### Initialize to calculate the distance between updates and global model
        target_params_variables = dict()
        for name, param in self.global_model.state_dict().items():
            target_params_variables[name] = param.clone()

        ### Start training for each participating local client
        aggregated_model_id = [1] * self.params["no_of_participants_per_round"]

        # ------ FedRecover Code Start ------
        # save global model
        if round == self.params["start_round"]:
            self.global_model_at_start = copy.deepcopy(self.global_model.state_dict())

        # save server model every round after pre_recover_rounds
        if round >= self.params["start_round"] and round <= self.recovering_end_round:
            self.server_model_history.append(copy.deepcopy(self.global_model.state_dict()))

        # at pre_recover_rounds, save the local model, because we calculate delta from there.
        if round >= self.params["start_round"] and round < self.pre_recover_rounds:
            for model_id in selected_clients:
                if model_id in self.malicious_clients:  # only record the local models after attack for malicious clients
                    pass
                else:  # for benign clients, record every round.
                    continue
            self.global_model_at_start = copy.deepcopy(self.global_model.state_dict())

        # Flag for recovery phase
        if round == self.recovering_start_round:
            self.recovering = True
            logger.info(f"Entering recovery phase at round {round}")
        elif round == self.recovering_end_round:
            self.recovering = False
            logger.info(f"Exiting recovery phase at round {round}")

        if self.recovering:
            logger.info(f"Recovering malicious clients {self.malicious_clients} updates at round {round}")

        # ------ FedRecover Code End ------

        for model_id in selected_clients:
            logger.info(f" ")
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = train_dataloader[model_id]
            else:
                client = local_benign_client
                client_train_data = train_dataloader[model_id]

            ### count class distribution info
            if self.params["show_local_test_log"]:
                distrib_dict, percentage_dict, sum_no = self.local_data_distrib(client_train_data)
                logger.info(f"class distribution for model {model_id}, total no:{sum_no}")
                logger.info(f"{distrib_dict}")
                logger.info(f"{percentage_dict}")

            ### copy global model
            client.local_model.copy_params(self.global_model.state_dict())

            ### set requires_grad to True
            for name, params in client.local_model.named_parameters():
                params.requires_grad = True

            client.local_model.train()
            start_time = time.time()
            client.local_training(
                train_data=client_train_data,
                target_params_variables=target_params_variables,
                test_data=test_dataloader,
                is_log_train=self.params["show_train_log"],
                poisoned_pattern_choose=self.params["poisoned_pattern_choose"],
                round=round, model_id=model_id, former_model=former_model
            )

            logger.info(f"local training for model {model_id} finishes in {time.time() - start_time} sec")

            if model_id in self.malicious_clients:
                logger.info(f"BEFORE clipping:")
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader,
                                  poisoned_pattern_choose=self.params["poisoned_pattern_choose"])
                logger.info(f" ")

            ### Clip the parameters norm to the agreed bound
            self._norm_clip(local_client=client, round=round, model_id=model_id)

            cs = self._cos_sim(client=client, target_params_variables=target_params_variables)
            logger.info(f"cosine similarity between model {model_id} and global model is {cs}")

            logger.info(f" ")
            # ------ Original Code End ------

            # ------ FedRecover Code Start ------

            local_update = {}
            for name, param in client.local_model.state_dict().items():
                local_update[name] = (param - self.global_model.state_dict()[name]).detach().cpu()  # save model update

            # Accumulate updates into weight_accumulator (this part is now correct for all clients)
            if self.recovering and model_id in self.malicious_clients:

                estimated_update = self.estimate_malicious_update(model_id, round, selected_clients, adversary_list,
                                                                  local_malicious_client)

                for name, update in estimated_update.items():
                    weight_accumulator[name].add_(update.long().cuda())
                logger.info(f"Replaced malicious client {model_id} update with estimated update.")
                local_update = estimated_update


            else:
                #  normal client, all updates here
                for name, update in local_update.items():
                    weight_accumulator[name].add_(update.cuda())

            # Save historical updates, only for malicious clients
            if model_id in self.malicious_clients:
                if model_id not in self.local_update_history:
                    self.local_update_history[model_id] = deque(maxlen=50)

                self.local_update_history[model_id].append(local_update)  # store local updates of malicious clients

        # ------ Original Code Start ------
        for ind, model_id in enumerate(selected_clients):
            if model_id in self.malicious_clients:
                if aggregated_model_id[ind] == 0:
                    self.no_detected_malicious += 1
                else:
                    self.no_undetected_malicious += 1
            else:
                if aggregated_model_id[ind] == 0:
                    self.no_misclassified_benign += 1
                else:
                    self.no_detected_benign += 1

        self.no_processed_malicious_clients += len([i for i in selected_clients if i in self.malicious_clients])
        self.no_processed_benign_clients += len([i for i in selected_clients if i not in self.malicious_clients])
        logger.info(f"aggregated_model:{aggregated_model_id}")
        logger.info(
            f"correctly detected malicious clients:{self.no_detected_malicious}/{self.no_processed_malicious_clients}, \
                    undetected malicious clients:{self.no_undetected_malicious}/{self.no_processed_malicious_clients}")
        logger.info(f"correctly detected benign clients:{self.no_detected_benign}/{self.no_processed_benign_clients}, \
                    misclassified benign clients:{self.no_misclassified_benign}/{self.no_processed_benign_clients}")
        # ------ FedRecover: Global Test Add ------
        if self.recovering:
            # acc, asr = self.global_test(test_dataloader, round, poisoned_pattern_choose=self.params["poisoned_pattern_choose"]) # 确保 poisoned_pattern_choose 参数传递正确
            # logger.info(f"Recovery finished. Global model accuracy: {acc:.2f}%, ASR: {asr:.2f}%")

            for malicious_id in self.malicious_clients:

                recovered_model_state = self.get_recovered_malicious_model(malicious_id)
                if recovered_model_state is not None:
                    # Load the recovered model state into a copy of local_malicious_client's model
                    temp_malicious_model = copy.deepcopy(local_malicious_client.local_model)
                    temp_malicious_model.load_state_dict(recovered_model_state)
                    temp_malicious_model.cuda()  # Move to GPU if available

                    malicious_acc, malicious_asr = self.Recover_test(test_dataloader, round, model=temp_malicious_model,
                                                                     poisoned_pattern_choose=self.params[
                                                                         "poisoned_pattern_choose"])  # 传递模型

                else:
                    logger.warning(f"Could not retrieve recovered model for malicious client {malicious_id}")

        return weight_accumulator, aggregated_model_id

    def vectorize_update(self, local_update):
        """
        将local update (dict) 转化为一个向量 (np.array).
        """
        vectorized_update = []
        for name, tensor in local_update.items():
            vectorized_update.append(tensor.cpu().numpy().flatten())  # 将 tensor 转换为 numpy 数组并展开
        return np.concatenate(vectorized_update)

    def devectorize_update(self, update_vector):
        """
        将update (vector) 转化为local_update (dict).
        """
        local_update = {}
        start = 0

        # get the model structure from one benign client
        client = self.benign_clients[0]  # get a benign client
        if not self.local_update_history:
            return local_update
        local_update_sample = self.local_update_history[self.malicious_clients[0]][
            0]  # get local update sample from one malicious client, after checking update exist

        for name, tensor in local_update_sample.items():
            size = tensor.numel()  # 计算 tensor 的元素数量
            end = start + size
            update_array = update_vector[start:end]  # slice the part
            update_tensor = torch.tensor(update_array).reshape(tensor.shape)  # 恢复形状
            local_update[name] = update_tensor
            start = end

        return local_update

    def estimate_malicious_update(self, malicious_client_id, round, selected_clients, adversary_list,
                                  local_malicious_client):

        if malicious_client_id not in self.local_update_history or not self.local_update_history[malicious_client_id]:
            print(f"No historical updates found for malicious client {malicious_client_id}.")
            return {}

        target_update = self.local_update_history[malicious_client_id][-1]

        if not target_update:
            print(f"Empty update for malicious client {malicious_client_id} at round {round}")
            return {}

        previous_estimated_updates = self.local_update_history[malicious_client_id][-2] if len(
            self.local_update_history[malicious_client_id]) > 1 else None

        estimated_updates = {
            name: (previous_estimated_updates[name].clone().detach().to(torch.float32).requires_grad_(True)
                   if previous_estimated_updates and name in previous_estimated_updates
                   else torch.zeros_like(target_update[name], dtype=torch.float32).requires_grad_(True))
            for name in target_update.keys()
        }

        w_current = self.global_model.state_dict()
        w_original = self.global_model_history
        delta_w = {name: w_current[name] - w_original[name] for name in w_current.keys()}

        original_update = {name: target_update[name].clone().detach() for name in target_update.keys()}

        if not hasattr(self, 'lbfgs_buffers'):
            self.lbfgs_buffers = {}

        if malicious_client_id not in self.lbfgs_buffers:
            self.lbfgs_buffers[malicious_client_id] = {'delta_W': deque(maxlen=2), 'delta_G': deque(maxlen=2)}

        lbfgs_buffer = self.lbfgs_buffers[malicious_client_id]

        def approximate_hessian_vector_product(v):
            delta_W_list = list(lbfgs_buffer['delta_W'])
            delta_G_list = list(lbfgs_buffer['delta_G'])
            num_stored = len(delta_W_list)

            if num_stored == 0:
                return v

            rho, alpha, q = [], [], v

            device = next(iter(self.global_model.parameters())).device

            for i in reversed(range(num_stored)):
                delta_W = delta_W_list[i]
                delta_G = delta_G_list[i]

                delta_W = {name: delta_W[name].to(device) for name in delta_W.keys()}
                delta_G = {name: delta_G[name].to(device) for name in delta_G.keys()}

                awt_ag = sum(torch.sum(delta_W[name] * delta_G[name]) for name in delta_W.keys())

                rho_i = 1.0 / (awt_ag + 1e-8)
                rho.append(rho_i)

                alpha_i = {name: rho_i * torch.sum(delta_W[name] * q[name]) for name in delta_W.keys()}
                alpha.append(alpha_i)

                for name in delta_W.keys():
                    q[name] = q[name].to(torch.float32)

                    alpha_i[name] = alpha_i[name].to(torch.float32)
                    delta_G[name] = delta_G[name].to(torch.float32)

                    q[name] -= alpha_i[name] * delta_G[name]

            gamma = 1.0
            for name in q.keys():
                q[name] *= gamma

            for i in range(num_stored):
                delta_W = delta_W_list[i]
                delta_G = delta_G_list[i]

                delta_W = {name: delta_W[name].to(device) for name in delta_W.keys()}
                delta_G = {name: delta_G[name].to(device) for name in delta_G.keys()}

                beta = {name: rho[num_stored - 1 - i] * torch.sum(delta_G[name] * q[name]) for name in delta_G.keys()}

                for name in delta_W.keys():
                    q[name] = q[name].to(torch.float32)

                    beta[name] = beta[name].to(torch.float32)
                    delta_W[name] = delta_W[name].to(torch.float32)
                    q[name] += (alpha[num_stored - 1 - i][name].to(torch.float32) - beta[name]) * delta_W[name]

            return q

        def loss_fn(estimated_update, original_update, hessian_vector_product):
            return sum(torch.sum((estimated_update[name] - original_update[name].to(estimated_update[name].device)
                                  - hessian_vector_product[name].to(estimated_update[name].device)) ** 2)
                       for name in original_update.keys())

        optimizer = optim.LBFGS([estimated_updates[name] for name in estimated_updates], lr=0.1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            hessian_vector_product = approximate_hessian_vector_product(delta_w)
            loss = loss_fn(estimated_updates, original_update, hessian_vector_product)
            loss.backward()
            return loss

        optimizer.step(closure)

        # 11. 更新 L-BFGS 缓冲区
        with torch.no_grad():
            delta_W = {name: estimated_updates[name] - original_update[name] for name in original_update.keys()}
            delta_G = {name: original_update[name] for name in original_update.keys()}
            lbfgs_buffer['delta_W'].append(delta_W)
            lbfgs_buffer['delta_G'].append(delta_G)

        return estimated_updates

    def get_recovered_malicious_model(self, malicious_id):
        """
        获取恢复后的恶意客户端模型.
        """
        if malicious_id not in self.local_update_history:
            logger.warning(f"No local update history found for malicious client {malicious_id}")
            return None

        # 获取恶意客户端最后一次更新的状态字典
        last_update = self.local_update_history[malicious_id][-1]

        # 创建恶意客户端模型的深拷贝
        recovered_model_state = copy.deepcopy(self.global_model.state_dict())

        # 将估计的更新应用到模型上
        for name, update in last_update.items():
            recovered_model_state[name] = recovered_model_state[name] + update.cuda()

        return recovered_model_state

    def _poisoned_batch_injection(self, batch, poisoned_pattern_choose=None, evaluation=False, model_id=None):
        r"""
        replace the poisoned batch with the oirginal batch
        """
        poisoned_batch = copy.deepcopy(batch)
        original_batch = copy.deepcopy(batch)
        poisoned_len = self.params["poisoned_len"] if not evaluation else len(poisoned_batch[0])
        if self.params["semantic"]:
            poison_images_list = copy.deepcopy(self.params["poison_images"])
            random.shuffle(poison_images_list)
            poison_images_test_list = copy.deepcopy(self.params["poison_images_test"])
            random.shuffle(poison_images_test_list)

        for pos in range(len(batch[0])):
            if pos < poisoned_len:
                if self.params["semantic"] and not self.params["edge_case"]:
                    if not evaluation:
                        poison_choice = poison_images_list[pos % len(self.params["poison_images"])]
                    else:
                        poison_choice = poison_images_test_list[pos % len(self.params["poison_images_test"])]
                    poisoned_batch[0][pos] = self.train_dataset[poison_choice][0]
                elif self.params["semantic"] and self.params["edge_case"]:
                    transform_edge_case = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                    if not evaluation:
                        poison_choice = random.choice(range(len(self.edge_case_train)))
                        poisoned_batch[0][pos] = transform_edge_case(self.edge_case_train[poison_choice])
                    else:
                        poison_choice = random.choice(range(len(self.edge_case_test)))
                        poisoned_batch[0][pos] = transform_edge_case(self.edge_case_test[poison_choice])

                elif (self.params["pixel_pattern"] and poisoned_pattern_choose != None):
                    if poisoned_pattern_choose == 10:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose,
                                                             blend_pattern=self.blend_pattern,
                                                             blend_alpha=self.params["blend_alpha"])
                    elif poisoned_pattern_choose == 1:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)
                    elif poisoned_pattern_choose == 20:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose,
                                                             evaluation=evaluation, model_id=model_id)

                poisoned_batch[1][pos] = self.params["poison_label_swap"]

        return poisoned_batch, original_batch

    def _global_test_sub(self, test_data, model=None, test_poisoned=False, poisoned_pattern_choose=None):
        r"""
        test benign acc on global model
        """
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        correct = 0

        dataset_size = len(test_data.dataset)
        data_iterator = test_data
        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):
                if test_poisoned:
                    batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose,
                                                                           evaluation=True)
                else:
                    batch = copy.deepcopy(batch)
                    original_batch = copy.deepcopy(batch)

                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                _, original_targets = original_batch
                original_targets = original_targets.cuda().detach().requires_grad_(False)

                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, acc)

    def _global_test_asr(self, test_data, model=None, poisoned_pattern_choose=None):
        r"""
        test attack success rate on global model
        """
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        correct = 0
        success = 0

        dataset_size = len(test_data.dataset)
        data_iterator = test_data
        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):
                batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose,
                                                                       evaluation=True)  # inject poison here
                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                _, original_targets = original_batch
                original_targets = original_targets.cuda().detach().requires_grad_(False)

                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]

                success += pred.eq(
                    targets.data.view_as(pred)).cpu().sum().item()  # how many poisoned image get attacked

        asr = 100.0 * (float(success) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, asr)

    def global_test(self, test_data, round, poisoned_pattern_choose=None, model=None):
        r"""
        global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, model=model, test_poisoned=False)
        loss_p, asr = self._global_test_asr(test_data, model=model, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(
            f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}, asr:{asr}, poisoned loss:{loss_p}")

        return (acc, asr)

    def Recover_test(self, test_data, round, poisoned_pattern_choose=None, model=None):
        r"""
        global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, model=model, test_poisoned=False)
        loss_p, asr = self._global_test_asr(test_data, model=model, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(
            f"Recovered model on round:{round} | benign acc:{acc}, benign loss:{loss}, asr:{asr}, poisoned loss:{loss_p}")

        return (acc, asr)

    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True
