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
from torchvision.utils import save_image
import os
logger = logging.getLogger("logger")



class DatasetDisstiallionServer(AbstractServer):
    
    def __init__(self, params, current_time, train_dataset, blend_pattern,
                 edge_case_train, edge_case_test):
        super(DatasetDisstiallionServer, self).__init__(params, current_time)
        self.train_dataset=train_dataset
        self.blend_pattern=blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

        self.no_detected_malicious = 0
        self.no_undetected_malicious = 0
        self.no_detected_benign = 0
        self.no_misclassified_benign = 0
        self.no_processed_malicious_clients = 0
        self.no_processed_benign_clients = 0
        self._create_check_model()

    def _create_check_model(self):
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
        
        self.check_model = check_model.cuda()
        return True

    def _select_clients(self, round):
        r"""
        randomly select participating clients for each round
        """
        adversary_list = [i for i in range(self.params["no_of_adversaries"])] \
                            if round in self.poisoned_rounds else []

        selected_clients = random.sample(range(self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"]) \
                if round not in self.poisoned_rounds else \
                adversary_list + random.sample(range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"]-self.params["no_of_adversaries"])
        return selected_clients, adversary_list

    def aggregation(self, weight_accumulator, aggregated_model_id,round):
        r"""
        aggregate all the updates model to generate a new global model
        """
        no_of_participants_this_round = len(aggregated_model_id)
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * \
                        (self.params["eta"] / no_of_participants_this_round)

            data = data.float()
            data.add_(update_per_layer)
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
                new_value = self.global_model.state_dict()[name] + (local_client.local_model.state_dict()[name] - self.global_model.state_dict()[name])/scale
                local_client.local_model.state_dict()[name].copy_(new_value)

        return True

    def local_data_distrib(self, train_data):
        ### Initialization
        distrib_dict=dict()
        no_class = 100 if self.params["dataset"].upper()=="CIFAR100" else 10 
        for label in range(no_class):
            distrib_dict[label]=0
        
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
        percentage_dict=dict()
        for key,value in distrib_dict.items():
            percentage_dict[key] = round(value/sum_no, 2)

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

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader, poison_train_dataloader,former_model):

        r"""
        Server broadcasts the global model to all participants.
        Every participants train its our local model and upload the weight difference to the server.
        The server then aggregate the changes in the weight_accumulator and return it.
        """
        ### Log info
        logger.info(f"Training on global round {round} begins")
            
        ### Count adversaries in one global round
        current_no_of_adversaries = 0
        selected_clients, adversary_list= self._select_clients(round)
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

        ### Start training for each participating local client
        aggregated_model_id = [1]*self.params["no_of_participants_per_round"]

        for model_id in selected_clients:
            logger.info(f" ")
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = poison_train_dataloader
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
                                 train_data = client_train_data, 
                                 target_params_variables = target_params_variables,
                                 test_data = test_dataloader,
                                 is_log_train = self.params["show_train_log"],
                                 poisoned_pattern_choose = self.params["poisoned_pattern_choose"],
                                 round=round, model_id=model_id,former_model=former_model
                                  )

            def tv_loss(img):
                """计算 Total Variation Loss，增强清晰度"""
                return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
                    torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

            ipc = 1  # 每个类别生成一张图片
            class_num = self.params["class_num"]

            # 初始化合成图片，每个类别1张
            image_syn = torch.randn(size=(class_num, 3, 32, 32), dtype=torch.float, requires_grad=True, device='cuda:0')
            label_syn = torch.arange(class_num, dtype=torch.long, device='cuda:0')  # 生成 0~(class_num-1) 的标签

            print('Initialize synthetic data from random noise')

            # 选择要匹配的模型层，例如 self.global_model 的某个高层
            feature_extractor = nn.Sequential(*list(self.global_model.children())[:-2])  # 例如 ResNet 的倒数第二层
            feature_extractor.eval()  # 只提取特征，不更新权重

            # 提取目标特征（可以从某个客户端的本地数据中提取）
            real_images, real_labels = self.get_real_data()  # 需要实现，获取训练数据
            real_features = feature_extractor(real_images).detach()  # 提取真实特征

            # **实验不同优化策略**
            use_sgd = False  # 设为 True 切换到 SGD

            if use_sgd:
                optimizer_img = torch.optim.SGD([image_syn], lr=0.1, momentum=0.9)
            else:
                optimizer_img = torch.optim.Adam([image_syn], lr=0.1)

            # 反演过程
            for epoch in range(1000):
                optimizer_img.zero_grad()

                # 计算合成图像的特征
                syn_features = feature_extractor(image_syn)

                # **主要损失：匹配特征**
                feature_loss = ((syn_features - real_features) ** 2).mean()

                # **TV Loss：增强图片清晰度**
                tv = tv_loss(image_syn)

                # **L2 正则化**
                reg_loss = (image_syn ** 2).mean()

                # **总损失**
                loss = feature_loss + 1e-4 * tv + 1e-4 * reg_loss

                loss.backward()
                optimizer_img.step()

            # 反归一化
            std = [0.2023, 0.1994, 0.2010]
            mean = [0.4914, 0.4822, 0.4465]
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(3):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
            image_syn_vis.clamp_(0, 1)

            # 保存图像，每类一张
            save_name = os.path.join('saved_models', 'feature_inversion.png')
            save_image(image_syn_vis, save_name, nrow=ipc)
            logger.info(f"local training for model {model_id} finishes in {time.time()-start_time} sec")

            if model_id==0:
                logger.info(f"BEFORE clipping:")
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader, poisoned_pattern_choose=self.params["poisoned_pattern_choose"])
                logger.info(f" ")

            ### Clip the parameters norm to the agreed bound
            self._norm_clip(local_client=client, round=round, model_id=model_id)

            if model_id==0:
                logger.info(f"AFTER clipping:")
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader, poisoned_pattern_choose=self.params["poisoned_pattern_choose"])
                logger.info(f" ")

            cs = self._cos_sim(client=client, target_params_variables=target_params_variables)
            logger.info(f"cosine similarity between model {model_id} and global model is {cs}")
 
            logger.info(f" ")
            for name, param in client.local_model.state_dict().items():
                weight_accumulator[name].add_(param - self.global_model.state_dict()[name])

        for ind, model_id in enumerate(selected_clients):
            if ind in adversary_list:
                if aggregated_model_id[ind] == 0:
                    self.no_detected_malicious+=1
                else:
                    self.no_undetected_malicious+=1
            else:
                if aggregated_model_id[ind] == 0:
                    self.no_misclassified_benign+=1
                else:
                    self.no_detected_benign+=1

        self.no_processed_malicious_clients +=  len(adversary_list)
        self.no_processed_benign_clients +=  len(selected_clients) - len(adversary_list)
        logger.info(f"aggregated_model:{aggregated_model_id}")
        logger.info(f"correctly detected malicious clients:{self.no_detected_malicious}/{self.no_processed_malicious_clients}, \
                    undetected malicious clients:{self.no_undetected_malicious}/{self.no_processed_malicious_clients}")
        logger.info(f"correctly detected benign clients:{self.no_detected_benign}/{self.no_processed_benign_clients}, \
                    misclassified benign clients:{self.no_misclassified_benign}/{self.no_processed_benign_clients}")

        return weight_accumulator, aggregated_model_id

    def get_gw(self,client_model,global_model):
        return [client_param.data - global_param.data
                for client_param, global_param in zip(client_model.parameters(), global_model.parameters())]




    def match_loss(self,gw_syn, gw_real, type):
        def distance_wb(gwr, gws):
            shape = gwr.shape
            if len(shape) == 4:  # conv, out*in*h*w
                gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
                gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
            elif len(shape) == 3:  # layernorm, C*h*w
                gwr = gwr.reshape(shape[0], shape[1] * shape[2])
                gws = gws.reshape(shape[0], shape[1] * shape[2])
            elif len(shape) == 2:  # linear, out*in
                tmp = 'do nothing'
            elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
                gwr = gwr.reshape(1, shape[0])
                gws = gws.reshape(1, shape[0])
                return torch.tensor(0, dtype=torch.float, device=gwr.device)

            dis_weight = torch.sum(
                1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
            dis = dis_weight
            return dis
        dis = torch.tensor(0.0).cuda()

        if type == 'ours':
                for ig in range(len(gw_real)):
                    gwr = gw_real[ig]
                    gws = gw_syn[ig]
                    dis += distance_wb(gwr, gws)

        elif type == 'mse':
                gw_real_vec = []
                gw_syn_vec = []
                for ig in range(len(gw_real)):
                    gw_real_vec.append(gw_real[ig].reshape((-1)))
                    gw_syn_vec.append(gw_syn[ig].reshape((-1)))
                gw_real_vec = torch.cat(gw_real_vec, dim=0)
                gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
                dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

        elif type == 'cos':
                gw_real_vec = []
                gw_syn_vec = []
                for ig in range(len(gw_real)):
                    gw_real_vec.append(gw_real[ig].reshape((-1)))
                    gw_syn_vec.append(gw_syn[ig].reshape((-1)))
                gw_real_vec = torch.cat(gw_real_vec, dim=0)
                gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
                dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)



        return dis

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
                    if poisoned_pattern_choose==10:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, blend_pattern=self.blend_pattern, blend_alpha=self.params["blend_alpha"])
                    elif poisoned_pattern_choose==1:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)
                    elif poisoned_pattern_choose==20:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, evaluation=evaluation, model_id=model_id)

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
                    batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
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
                # if test_watermarking and batch_id==0:
                #     logger.info(f"targets:{targets}")
                #     logger.info(f"original targets:{original_targets}")
                #     logger.info(f"pred:{pred}")

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, acc)
    
    def global_test(self, test_data, round, poisoned_pattern_choose=None):
        r"""
        global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, test_poisoned = False)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned = True, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)
    
    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True