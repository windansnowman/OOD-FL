import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer
import numpy as np
import random
import logging
import time
import copy
import models.resnet
import models.vgg
from utils.utils import add_trigger
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator
from sklearn.decomposition._pca import PCA
from sklearn.utils import *
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector
logger = logging.getLogger("logger")


class Node(object):
    def __init__(self, index, lchild=None, rchild=None, distances=None, counts=None):
        self.index = index  # 创建节点值
        self.lchild = lchild  # 创建左子树
        self.rchild = rchild  # 创建右子树
        self.distances = distances  # 两个子树直接的距离
        self.counts = counts  # 包含叶子节点数量
        self.leaves = []

    def postorder_travel(self, node):
        # 如果节点为空
        if node == None:
            return []

        self.postorder_travel(node.lchild)
        self.postorder_travel(node.rchild)
        if node.counts == 1:
            self.leaves.append(node.index)
        return self.leaves
class FedDMCServer(AbstractServer):

    def __init__(self, params, current_time, train_dataset, blend_pattern,
                 edge_case_train, edge_case_test):
        super(FedDMCServer, self).__init__(params, current_time)
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
            adversary_list + random.sample(
                range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"] - self.params["no_of_adversaries"])
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
        return True

    def _norm_check(self, local_client, round, model_id):
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

    def _norm_clip(self, local_model_vector, clip_value):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_model_vector.items():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        scale = max(1.0, float(torch.abs(l2_norm / clip_value)))

        if self.params["norm_clip"]:
            for name, data in local_model_vector.items():
                new_value = self.global_model.state_dict()[name] + (
                            local_model_vector[name] - self.global_model.state_dict()[name]) / scale
                local_model_vector[name].copy_(new_value)

        return local_model_vector

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

    def agg_pca_agglomer(self,update_params, pca_d):
        user_grads = []
        for local_parameters in update_params:
            # --- 新增：归一化 ---
            # 计算 L2 范数（即向量长度）
            norm = torch.norm(local_parameters)
            # 避免除以零
            if norm > 0:
                normalized_params = local_parameters / norm
            else:
                normalized_params = local_parameters

            processed_params = normalized_params

            if len(user_grads) == 0:
                user_grads = processed_params.unsqueeze(0)
            else:
                user_grads = torch.cat(
                    (user_grads, processed_params.unsqueeze(0)), 0
                )
        param = user_grads.detach().cpu().numpy()
        param, _ = self.PCA_skl(param, pca_d)

        agglomer = AgglomerativeClustering(n_clusters=2, linkage='ward', compute_distances=True).fit(param)

        linkage_matrix = self.get_linkage_matrix(agglomer)
        tree = self.Building_tree(linkage_matrix, len(agglomer.labels_))
        min_cluster_size = 3
        benign, malicious, outlier_all = self.Removing_outliers(tree, min_cluster_size)
        benign_list=[]
        for value in benign:
            benign_list.append(int(value))

        return benign_list

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader,
                         poison_train_dataloader, former_model=None):

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
        aggregated_model_id = [0] * self.params["no_of_participants_per_round"]

        local_model_vector = []
        local_model_named_param = []
        update_params = []
        local_model_state_dict = []
        for model_id in selected_clients:
            logger.info(f" ")
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = train_dataloader[model_id]
                # client_train_data = poison_train_dataloader
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
                round=round, model_id=model_id
            )

            logger.info(f"local training for model {model_id} finishes in {time.time() - start_time} sec")

            if model_id == 0:
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader,
                                  poisoned_pattern_choose=self.params["poisoned_pattern_choose"])
                logger.info(f" ")

            ### Clip the parameters norm to the agreed bound
            self._norm_check(local_client=client, round=round, model_id=model_id)

            cs = self._cos_sim(client=client, target_params_variables=target_params_variables)
            logger.info(f"cosine similarity between model {model_id} and global model is {cs}")

            update_params_sub = []
            for name, param in client.local_model.named_parameters():
                update_params_value = param.clone() - target_params_variables[name].clone()
                update_params_sub.append(update_params_value.view(-1))
            update_params_sub = torch.cat(update_params_sub).cuda()
            update_params.append(update_params_sub)

            local_model_state_dict_sub = dict()
            for name, param in client.local_model.state_dict().items():
                local_model_state_dict_sub[name] = param.clone()
            local_model_state_dict.append(local_model_state_dict_sub)

            ### Updates the weight accumulator
            # for name, param in client.local_model.state_dict().items():
            #     weight_accumulator[name].add_(param -
            #     self.global_model.state_dict()[name])

        logger.info(f" ")
        benign_client = self.agg_pca_agglomer(update_params=update_params, pca_d=10)
        # logger.info(f"benign clients are:{benign_client}")
        for ind in benign_client:
            aggregated_model_id[ind] = 1
            for name, param in local_model_state_dict[ind].items():
                weight_accumulator[name].add_(param - self.global_model.state_dict()[name])

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

        return weight_accumulator, aggregated_model_id

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
        loss, acc = self._global_test_sub(test_data, test_poisoned=False)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned=True,
                                              poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)

    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True





    def Building_tree(self,linkage_matrix, n_samples):
        cluster_id = n_samples
        queue = {}
        root = None

        for child in linkage_matrix:
            if child[0] < n_samples:
                lchild = Node(child[0], counts=1)
            else:
                lchild = queue[child[0]]
                del queue[child[0]]

            if child[1] < n_samples:
                rchild = Node(child[1], counts=1)
            else:
                rchild = queue[child[1]]
                del queue[child[1]]

            root = Node(cluster_id, lchild, rchild, child[2], child[3])
            queue[cluster_id] = root
            cluster_id = cluster_id + 1
        return root

    def Removing_outliers(self,root, min_cluster_size=3):
        outlier_all = []
        n_clients = root.counts

        while root.rchild.counts <= min_cluster_size or root.lchild.counts <= min_cluster_size:

            if root.rchild.counts >= min_cluster_size:

                # outlier = root.lchild.preorder()
                outlier = root.lchild.postorder_travel(root.lchild)
                root = root.rchild

            elif root.lchild.counts >= min_cluster_size:
                # 如果左孩子的叶子节点数量满足，则右孩子的叶子节点标位-1，然后从树中删除。
                outlier = root.rchild.postorder_travel(root.rchild)
                root = root.lchild

            else:
                outlier = root.postorder_travel(root)
                outlier_all.extend(outlier)
                root = None
                break
            outlier_all.extend(outlier)
            if len(outlier_all) > (n_clients // 2):
                break

        # root中，左右孩子分别是良性和恶意用户
        benign, malicious = [], []
        if root:
            if root.lchild.counts > (n_clients // 2) or root.rchild.counts > (n_clients // 2):
                if root.rchild.counts < root.lchild.counts:
                    malicious = root.rchild.postorder_travel(root.rchild)
                    benign = root.lchild.postorder_travel(root.lchild)
                elif root.rchild.counts > root.lchild.counts:
                    benign = root.rchild.postorder_travel(root.rchild)
                    malicious = root.lchild.postorder_travel(root.lchild)
            else:
                benign = root.postorder_travel(root)
        return benign, malicious, outlier_all

    def get_linkage_matrix(self,agglomer):
        counts = np.zeros(agglomer.children_.shape[0])
        n_samples = len(agglomer.labels_)

        for i, merge in enumerate(agglomer.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        d = agglomer.distances_

        child = agglomer.children_
        linkage_matrix = np.column_stack([agglomer.children_, d, counts]).astype(float)

        return linkage_matrix


    def PCA_skl(self,X, n_components=2, random_state=0):
        # base = BaseEstimator()
        X = BaseEstimator()._validate_data(X, accept_sparse=['csr'], ensure_min_samples=2,
                                           dtype=[np.float32, np.float64])
        random_state = check_random_state(random_state)
        pca = PCA(n_components=n_components, svd_solver='randomized',
                  random_state=random_state)
        # print(pca.explained_variance_ratio_)
        X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        u = pca.fit_transform(X)
        return X_embedded, u


