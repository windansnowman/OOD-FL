import torch
import torch.nn as nn
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer
import numpy as np
import logging
import time
import copy
import math
import random
import models.resnet
import models.vgg
from utils.utils import add_trigger
import models.cnn

logger = logging.getLogger("logger")


class OursServer(AbstractServer):

    def __init__(self, params, current_time, train_dataset, open_set, id_data,
                 blend_pattern, edge_case_train, edge_case_test, test_dataset, open_set_label=None):
        super(OursServer, self).__init__(params, current_time)
        self.defense_rounds = [round for round in range(self.params["defense_start_round"],
                                                        self.params["defense_end_round"],
                                                        self.params["defense_round_interval"])]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.open_set = open_set
        self.open_set_label = open_set_label
        self.blend_pattern = blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test
        self.id_data = id_data
        self.poisoned_acc = []
        self.clean_acc = []
        self.no_detected_malicious = 0
        self.no_undetected_malicious = 0
        self.no_detected_benign = 0
        self.no_misclassified_benign = 0
        self.no_processed_malicious_clients = 0
        self.no_processed_benign_clients = 0
        self.detection_alpha = self.params["detection_alpha"]
        self.wm_mu = self.params["balance_mu"]

        self._create_additional_model()
        self._loss_function()
        self.before_wm_injection_bn_stats_dict = dict()
        self.after_wm_injection_bn_stats_dict = dict()

    def _create_additional_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")
            elif self.params["dataset"].upper() == "TINY-IMAGENET":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=200, dataset="CIFAR")
        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)

        elif "CNN" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.cnn, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.cnn, self.params["model_type"])(num_classes=100)
        self.check_model = check_model.cuda()
        return True

    def _select_clients(self, round):
        r"""
        randomly select participating clients for each round
        """
        adversary_list = [i for i in range(self.params["no_of_adversaries"])] \
            if round in self.poisoned_rounds else []
        # adversary_list=[51]
        selected_clients = random.sample(range(self.params["no_of_total_participants"]), \
                                         self.params["no_of_participants_per_round"]) \
            if round not in self.poisoned_rounds else \
            adversary_list + random.sample(
                range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"] - self.params["no_of_adversaries"])
        return selected_clients, adversary_list
    def select_clients_for_round(self, round):
        """
        Selects clients for a given round. Adversaries are included every 10 rounds.
        """
        # Determine if the current round is an adversarial round (every 10th round)
        is_adversarial_round = (round % 10 == 0)

        # Define the list of potential adversaries (indices 0 to no_of_adversaries-1)
        all_adversary_indices = list(range(self.params["no_of_adversaries"]))

        if is_adversarial_round:
            # Adversarial Round (e.g., 0, 10, 20...)
            adversary_list = all_adversary_indices  # All adversaries participate
            num_adversaries = len(adversary_list)

            # Calculate how many benign clients are needed
            num_benign_needed = self.params["no_of_participants_per_round"] - num_adversaries

            # --- Input Validation (Good Practice) ---
            if num_benign_needed < 0:
                raise ValueError(
                    f"Round {round}: Cannot select {self.params['no_of_participants_per_round']} participants "
                    f"as it's less than the number of adversaries ({num_adversaries}).")

            # Define the pool of benign clients (indices from no_of_adversaries onwards)
            benign_client_pool = list(range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]))
            num_benign_available = len(benign_client_pool)

            if num_benign_available < num_benign_needed:
                raise ValueError(f"Round {round}: Not enough benign clients available. "
                                 f"Need {num_benign_needed}, but only {num_benign_available} exist "
                                 f"(range {self.params['no_of_adversaries']} to {self.params['no_of_total_participants'] - 1}).")
            # --- End Validation ---

            # Sample the required number of benign clients
            selected_benign = random.sample(benign_client_pool, num_benign_needed)

            # Combine adversaries and selected benign clients
            selected_clients = adversary_list + selected_benign

            # Optional: Shuffle the list if the order shouldn't reveal who is an adversary
            # random.shuffle(selected_clients)

        else:
            # Normal (Non-Adversarial) Round
            adversary_list = []  # No adversaries participate in this round

            # Sample randomly from the *entire* pool of participants
            total_client_pool = list(range(self.params["no_of_total_participants"]))

            # --- Input Validation ---
            if len(total_client_pool) < self.params["no_of_participants_per_round"]:
                raise ValueError(f"Round {round}: Not enough total clients ({len(total_client_pool)}) "
                                 f"to select {self.params['no_of_participants_per_round']}.")
            # --- End Validation ---

            selected_clients = random.sample(total_client_pool,
                                             self.params["no_of_participants_per_round"])

        return selected_clients, adversary_list

    def aggregation(self, weight_accumulator, aggregated_model_id, round):
        r"""
        aggregate all the updates model to generate a new global model
        """
        no_of_participants_this_round = sum(aggregated_model_id)
        if self.params['Adaptive_prune']:
            no_of_participants_this_round = self.params["no_of_participants_per_round"]
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * \
                               (self.params["eta"] / no_of_participants_this_round)

            data = data.float()
            data.add_(update_per_layer)

        return True

    def _check_norm(self, local_client, round, model_id):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_client.local_model.named_parameters():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)
        logger.info(f"round:{round}, local model {model_id} | l2_norm: {l2_norm}")

        return True

    def _norm_clip(self, local_model_state_dict, clip_value):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_model_state_dict.items():
            if "running" in name or "num_batches_tracked" in name:
                continue
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        scale = max(1.0, float(torch.abs(l2_norm / clip_value)))

        if self.params["norm_clip"]:
            for name, data in local_model_state_dict.items():
                if "running" in name or "num_batches_tracked" in name:
                    continue
                new_value = self.global_model.state_dict()[name] + (
                        local_model_state_dict[name] - self.global_model.state_dict()[name]) / scale
                local_model_state_dict[name].copy_(new_value)

        return local_model_state_dict

    def Bias_computation(self, local_model_state_dict, ood=True):
        score_list = []
        test_data = self.open_set if ood else self.id_data
        for ind, model_state_dict in enumerate(local_model_state_dict):

            self.check_model.copy_params(self.global_model.state_dict())
            for name, data in model_state_dict.items():
                if "num_batches_tracked" in name:
                    continue
                new_value = data.clone().detach()
                if ood:
                    if "running" in name:
                        if self.params["replace_original_bn"]:
                            new_value = self.after_wm_injection_bn_stats_dict[name]
                        else:
                            continue
                self.check_model.state_dict()[name].copy_(new_value)

            wm_copy_data = copy.deepcopy(test_data)
            score = self._global_Bias_test_sub(test_data=wm_copy_data, model=self.check_model)
            score_list.append(score)
        return score_list

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader,
                         poison_train_dataloader, former_model):
        r"""
        Server broadcasts the global model to all participants.
        Every participants train its our local model and upload the weight difference to the server.
        The server then aggregate the changes in the weight_accumulator and return it.
        """
        ### Log info
        logger.info(f"Training on global round {round} begins")
        self.test_dataloader = test_dataloader
        ### Count adversaries in one global round
        current_no_of_adversaries = 0
        selected_clients, adversary_list = self._select_clients(round)
        for client_id in selected_clients:
            if client_id in adversary_list:
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

        local_model_state_dict = []
        local_norm_list = []
        self.poison_train_dataloader = poison_train_dataloader
        for enum_id, model_id in enumerate(selected_clients):
            logger.info(f" ")

            if model_id in adversary_list:
                self.client = local_malicious_client
                client_train_data = poison_train_dataloader
                # client_train_data = train_dataloader[model_id]
            else:
                self.client = local_benign_client
                client_train_data = train_dataloader[model_id]

            ### copy global model
            self.client.local_model.copy_params(self.global_model.state_dict())

            ### set requires_grad to True
            for name, params in self.client.local_model.named_parameters():
                params.requires_grad = True

            self.client.local_model.train()
            start_time = time.time()
            self.client.local_training(
                train_data=client_train_data,
                target_params_variables=target_params_variables,
                test_data=test_dataloader,
                is_log_train=self.params["show_train_log"],
                poisoned_pattern_choose=self.params["poisoned_pattern_choose"],
                round=round, model_id=model_id, former_model=former_model
            )

            logger.info(f"local training for model {model_id} finishes in {time.time() - start_time} sec")

            if model_id in adversary_list:
                self.client.local_test(round=round, model_id=model_id, test_data=test_dataloader,
                                       poisoned_pattern_choose=self.params['poisoned_pattern_choose'])
                logger.info(f" ")

            ### Clip the parameters norm to the agreed bound
            self._check_norm(local_client=self.client, round=round, model_id=model_id)
            norm = self._model_dist_norm(model=self.client.local_model,
                                         target_params=target_params_variables)
            local_norm_list.append(norm)

            local_model_state_dict_sub = dict()
            for name, param in self.client.local_model.state_dict().items():
                local_model_state_dict_sub[name] = param.clone().detach()
            local_model_state_dict.append(local_model_state_dict_sub)

        if round in self.defense_rounds:
            id_score = self.Bias_computation(local_model_state_dict=local_model_state_dict, ood=False)
            ood_score = self.Bias_computation(local_model_state_dict=local_model_state_dict, ood=True)
            benign_client, malicious_client, target_label = self.detection(ood_score, id_score)
            # benign_client=[1,2,3,4,5,6,7,8,9]
            # malicious_client=[0]
        else:
            benign_client = [i for i in range(len(selected_clients))]
            malicious_client = []

        local_norm_list = np.array(local_norm_list)

        clip_value = np.median(local_norm_list[benign_client]) \
            if not self.params["fix_nc_bound"] else self.params["nc_bound"]

        logger.info(f" ")
        if self.params["norm_clip"]:
            logger.info(f"Norm clip: clipped value is: {clip_value}")
        else:
            logger.info(f"Norm clip: dont clip in this round")

        aggregated_model_id = [0] * len(local_model_state_dict)

        ### Updates the weight accumulator
        for ind in benign_client:
            aggregated_model_id[ind] = 1

            local_model_state_dict_clipped = self._norm_clip(local_model_state_dict[ind], clip_value)
            for name, param in local_model_state_dict[ind].items():
                if "num_batches_tracked" in name:
                    continue
                weight_accumulator[name].add_(
                    local_model_state_dict_clipped[name] - self.global_model.state_dict()[name])
        if self.params['Adaptive_prune']:
            for num, ind in enumerate(malicious_client):

                after_pruned_model, local_model_state_dict_pruned = self.Adaptive_prune(local_model_state_dict[ind],
                                                                                        self.open_set,
                                                                                        target_label[num])
                local_malicious_client.local_test(round=round, model_id='afetr_pruned', test_data=test_dataloader,
                                                  poisoned_pattern_choose=self.params['poisoned_pattern_choose'],
                                                  model=after_pruned_model)
                local_model_state_dict_clipped = self._norm_clip(local_model_state_dict_pruned, clip_value)
                for name, param in local_model_state_dict[ind].items():
                    if "num_batches_tracked" in name:
                        continue
                    weight_accumulator[name].add_(
                        torch.where(
                            local_model_state_dict_clipped[name] == 0,
                            self.global_model.state_dict()[name],
                            local_model_state_dict_clipped[name]) - self.global_model.state_dict()[name])
        for ind, model_id in enumerate(selected_clients):
            if ind in adversary_list:
                if aggregated_model_id[ind] == 0:
                    self.no_detected_malicious += 1
                else:
                    self.no_undetected_malicious += 1
            else:
                if aggregated_model_id[ind] == 0:
                    self.no_misclassified_benign += 1
                else:
                    self.no_detected_benign += 1

        self.no_processed_malicious_clients += len(adversary_list)
        self.no_processed_benign_clients += len(selected_clients) - len(adversary_list)
        logger.info(f"aggregated_model:{aggregated_model_id}")
        logger.info(
            f"correctly detected malicious clients:{self.no_detected_malicious}/{self.no_processed_malicious_clients}, \
                    undetected malicious clients:{self.no_undetected_malicious}/{self.no_processed_malicious_clients}")
        logger.info(f"correctly detected benign clients:{self.no_detected_benign}/{self.no_processed_benign_clients}, \
                    misclassified benign clients:{self.no_misclassified_benign}/{self.no_processed_benign_clients}")
        logger.info(f"current detection alpha: {self.detection_alpha}")

        return weight_accumulator, aggregated_model_id

    def detection(self, ood_score, id_score):
        malicious_client = []
        benign_client = []
        target_label = []
        out_list = []
        loc_list = []
        for ind in range(len(ood_score)):
            IO_shift = np.abs(ood_score[ind] - id_score[ind])
            upper_bound = self.detection_alpha
            loc = np.argmax(IO_shift)
            outlier_positions = IO_shift[loc]
            out_list.append(float(outlier_positions))
            loc_list.append(loc)
            if outlier_positions > upper_bound:
                malicious_client.append(ind)
                target_label.append(loc)
            else:
                benign_client.append(ind)
        logger.info(f'label :{loc_list}')
        logger.info(f'IO shift score :{out_list}')
        return benign_client, malicious_client, target_label

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

    def prune_step(self, ood_dataset, model_state, target_label, ratio):
        ceriterion = nn.CrossEntropyLoss()
        grad_ood = []
        ood_data = copy.deepcopy(ood_dataset)
        global_dict = self.global_model.state_dict()
        self.check_model.eval()
        for name_model, data_model in model_state.items():
            if "num_batches_tracked" in name_model:
                continue
            else:
                new_value = data_model.clone().detach()

            self.check_model.state_dict()[name_model].copy_(new_value)
        ood_model = copy.deepcopy(self.check_model)
        for inputs, labels in ood_data:
            labels[:] = torch.tensor(target_label)
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = ood_model(inputs)
            loss_ood = ceriterion(outputs, labels)
            loss_ood.backward()
        for _, params in ood_model.named_parameters():
            if "num_batches_tracked" in _:
                continue
            elif params.requires_grad and len(params.shape) > 1:
                grad_ood.append((params.grad.data ** 2).view(-1))
        grad_list_ood = torch.cat(grad_ood).cuda()
        ratio_prune = 0
        for step in range(100):
            _, indices = torch.topk(grad_list_ood, int(len(grad_list_ood) * ratio_prune))
            ratio_prune += ratio
            mask = torch.ones(len(grad_list_ood)).cuda()
            mask[indices] = 0
            count = 0
            for key, w in self.check_model.named_parameters():
                if "num_batches_tracked" in key:
                    continue
                elif w.requires_grad and len(w.shape) > 1:
                    after_prune = mask[count:count + len(w.data.view(-1))]
                    new_value = after_prune.reshape(w.data.size()) * (w.data - global_dict[key]) + global_dict[key]
                    self.check_model.state_dict()[key].copy_(new_value)
                    count += len(w.view(-1))
            ood_test = copy.deepcopy(self.check_model)
            for name_model, data_model in self.check_model.state_dict().items():
                if "num_batches_tracked" in name_model:
                    continue
                if "running" in name_model:
                    if self.params["replace_original_bn"]:
                        new_value = self.after_wm_injection_bn_stats_dict[name_model]
                    else:
                        continue
                else:
                    new_value = data_model.clone().detach()

                ood_test.state_dict()[name_model].copy_(new_value)
            after_prune_ratio = self.prune_test(test_data=ood_dataset, model=ood_test)
            # print()
            if after_prune_ratio[target_label] < 0.5:
                break

    def Adaptive_prune(self, model_state, ood_data, target_label):
        ratio = 0.01
        self.prune_step(ood_data, model_state, target_label, ratio)
        local_model_state_dict_sub = dict()
        for name, param in self.check_model.state_dict().items():
            local_model_state_dict_sub[name] = param.clone().detach()
        out_model = copy.deepcopy(self.check_model)
        return out_model, local_model_state_dict_sub

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
                    test_batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose,
                                                                                evaluation=True)
                else:
                    test_batch = copy.deepcopy(batch)
                    original_batch = copy.deepcopy(batch)

                data, targets = test_batch
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
        if test_poisoned:
            self.poisoned_acc.append(acc)
        else:
            self.clean_acc.append(acc)
        return (total_l, acc)

    def prune_test(self, test_data, model=None):
        if model == None:
            model = self.global_model
        model.eval()
        total_loss = 0
        dataset_size = 0
        data_iterator = copy.deepcopy(test_data)
        class_probs_sum = np.zeros(self.params['class_num'])
        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):
                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)
                output = model(data)
                total_loss += self.ceriterion(output, targets, reduction='sum').item()
                dataset_size += len(targets)
                probs = torch.softmax(output, dim=1)
                class_probs_sum += probs.sum(dim=0).detach().cpu().numpy()
        class_probs_sum = class_probs_sum / dataset_size
        model.train()
        return class_probs_sum

    def _global_Bias_test_sub(self, test_data, model=None):
        if model == None:
            model = self.global_model
        model.eval()
        total_loss = 0
        dataset_size = 0
        data_iterator = test_data
        class_probs_sum = np.zeros(self.params['class_num'])
        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):
                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)
                output = model(data)
                total_loss += self.ceriterion(output, targets, reduction='sum').item()
                dataset_size += len(targets)
                probs = torch.softmax(output, dim=1)
                class_probs_sum += probs.sum(dim=0).detach().cpu().numpy()
        class_probs_sum = class_probs_sum / dataset_size
        model.train()
        return class_probs_sum

    def global_test(self, test_data, round, poisoned_pattern_choose=None, model=None):
        r"""
        global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, test_poisoned=False, model=model)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned=True,
                                              poisoned_pattern_choose=poisoned_pattern_choose, model=model)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)

    def ceriterion_build(self, input, target, reduction=None):
        loss = nn.functional.cross_entropy(input, target, reduction=reduction)
        return loss

    def _loss_function(self):
        self.ceriterion = self.ceriterion_build
        return True

    def _optimizer(self, round, model):
        lr = self.params["global_lr"]
        momentum = self.params["global_momentum"]
        weight_decay = self.params["global_weight_decay"]

        # logger.info(f"indicator lr:{lr}")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        return True

    def _scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=self.params['global_milestones'],
                                                              gamma=self.params['global_lr_gamma'])
        return True

    def _model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    def _model_dist_norm_var(self, model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def _Simulate_OOD_BN(self, watermark_data, target_params_variables, round=None, model=None):

        if model == None:
            model = self.global_model
        model.train()

        total_loss = 0
        self._loss_function()
        self._optimizer(round, model)
        self._scheduler()

        retrain_no_times = self.params["global_retrain_no_times"]

        for internal_round in range(retrain_no_times):

            data_iterator = copy.deepcopy(watermark_data)

            for batch_id, watermark_batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                wm_data, wm_targets = watermark_batch
                wm_data = wm_data.cuda().detach().requires_grad_(False)
                wm_targets = wm_targets.cuda().detach().requires_grad_(False)

                data = wm_data
                targets = wm_targets

                output = model(data)

                class_loss = nn.functional.cross_entropy(output, targets)
                distance_loss = self._model_dist_norm_var(model, target_params_variables)
                loss = class_loss + (self.wm_mu / 2) * distance_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.data

            self.scheduler.step()

        return True

    def pre_process(self, test_data, round):
        copy_model = copy.deepcopy(self.global_model)
        ### Initialize to calculate the distance between updates and global model
        if round in self.defense_rounds:
            target_params_variables = dict()
            for name, param in self.global_model.state_dict().items():
                target_params_variables[name] = param.clone()

            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.before_wm_injection_bn_stats_dict[key] = value.clone().detach()

            wm_data = copy.deepcopy(self.open_set)
            self._Simulate_OOD_BN(watermark_data=wm_data,
                                  target_params_variables=target_params_variables,
                                  model=copy_model,
                                  round=round
                                  )

            for key, value in copy_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.after_wm_injection_bn_stats_dict[key] = value.clone().detach()

            self.check_model.copy_params(copy_model.state_dict())
            for key, value in self.check_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.check_model.state_dict()[key]. \
                        copy_(self.before_wm_injection_bn_stats_dict[key])
        return True

    def post_process(self):
        return True
