import torch
import torch.utils.data
from torchvision import datasets, transforms
from collections import Counter
from dataloader.AbstractDataloader import AbstractDataloader
import random
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
import pickle
import os
from utils.utils import NoiseDataset, RandomImages
from sklearn import preprocessing
from torch.utils.data import Dataset
from typing import Any
import glob
from PIL import Image
logger = logging.getLogger("logger")

class WMFLDataloader(AbstractDataloader):
    def __init__(self, params):
        super(WMFLDataloader, self).__init__(params)
        self.load_dataset()
        self.create_loader()

    def load_dataset(self):
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_ood = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_emnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        transform_train_Imagenet = transforms.Compose([
            # transforms.RandomCrop(64, padding=8),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test_Imagenet = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_ood_Imagenet = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])



        if self.params["dataset"].upper() == "CIFAR10":
            self.train_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR10("./data", train=False, download=True,
                                                 transform=transform_test)
            self.ood_dataset1 = datasets.CIFAR100("./data", train=True, download=True,
                                                  transform=transform_ood)
            if self.params["ood_data_source"] == "CIFAR100":
                self.ood_dataset = datasets.CIFAR100("./data", train=True, download=True, 
                                                  transform=transform_ood)
            elif self.params["ood_data_source"] == "EMNIST":
                self.ood_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            elif self.params["ood_data_source"] == "300KRANDOM":
                self.ood_dataset = RandomImages(transform=transform_ood, data_num=self.params["ood_data_sample_lens"])
            elif self.params["ood_data_source"] == "NOISE":
                self.ood_dataset = NoiseDataset(size=(3,32,32), num_samples=self.params["ood_data_sample_lens"])
            elif self.params["ood_data_source"] == "GTSRB":
                self.ood_dataset = datasets.GTSRB("./data",split='train',download=True,transform=transform_ood)
            elif self.params["ood_data_source"] == "CIFAR10":
                self.ood_dataset = datasets.CIFAR10("./data", train=False, download=True,
                                                  transform=transform_ood)

        elif self.params["dataset"].upper() == "CIFAR100":
            self.train_dataset = datasets.CIFAR100("./data", train=True, download=True, 
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR100("./data", train=False, download=True,
                                                 transform=transform_test)
            self.ood_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                  transform=transform_ood)
        elif self.params["dataset"].upper() == "EMNIST":
            self.train_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            self.test_dataset = datasets.EMNIST("./data", train=False, split="mnist", transform=transform_emnist)
            self.ood_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                  transform=transform_ood)

        elif self.params["dataset"].upper() == "TINY-IMAGENET":
            train_dir = 'data/tiny-imagenet-200/train'
            self.train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train_Imagenet)
            root = 'data/tiny-imagenet-200'
            id_dic = {}
            with open(os.path.join(root, 'wnids.txt'), 'r') as file:
                for i, line in enumerate(file):
                    id_dic[line.strip()] = i  # Use strip() to remove newline characters

            try:
                self.train_dataset = torch.load(root+'train_dataset.pt')
                self.test_dataset = torch.load(root+'val_dataset.pt')
                print("Loading a saved dataset")
            except FileNotFoundError:
                print("The saved dataset was not found, reload and save")

                self.save_tiny_dataset(root, id_dic, transform_train_Imagenet, transform_test_Imagenet)
                self.train_dataset = torch.load(root+'train_dataset.pt')
                self.test_dataset = torch.load(root+'val_dataset.pt')

            if self.params["ood_data_source"] == "CIFAR100":
                self.ood_dataset = datasets.CIFAR100("./data", train=True, download=True,
                                                     transform=transform_ood_Imagenet)
            elif self.params["ood_data_source"] == "EMNIST":
                self.ood_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                                   transform=transform_ood_Imagenet)
            elif self.params["ood_data_source"] == "300KRANDOM":
                self.ood_dataset = RandomImages(transform=transform_ood_Imagenet, data_num=self.params["ood_data_sample_lens"])
            elif self.params["ood_data_source"] == "NOISE":
                self.ood_dataset = NoiseDataset(size=(3, 64, 64), num_samples=self.params["ood_data_sample_lens"])
            elif self.params["ood_data_source"] == "GTSRB":
                self.ood_dataset = datasets.GTSRB("./data", split='train', download=True, transform=transform_ood_Imagenet)

        return True

    def _sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}

        # 按标签对数据集进行分组
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if self.params["semantic"] and (
                    ind in self.params['poison_images'] or ind in self.params['poison_images_test']):
                continue

            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        # 按照Dirichlet分布将样本分配给每个客户端
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))

            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        # 全局打乱每个客户端的数据
        shuffled_participants_list = list(per_participant_list.items())
        random.shuffle(shuffled_participants_list)

        # 将打乱后的数据重新赋值给 per_participant_list
        per_participant_list = defaultdict(list)
        for user, data in shuffled_participants_list:
            per_participant_list[user] = data

        return per_participant_list
    
    def _load_edge_case(self):
        with open('./data/edge-case/southwest_images_new_train.pkl', 'rb') as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)
        with open('./data/edge-case/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)        

        return saved_southwest_dataset_train, saved_southwest_dataset_test

    def _get_poison_train(self):
        indices = list()
        range_no_id = list(range(50000))
        for image in self.params['poison_images'] + self.params['poison_images_test']:
            if image in range_no_id and self.params['semantic']:
                range_no_id.remove(image)
        
        # add random images to other parts of the batch
        for batches in range(self.params["poison_no_reuse"]):
            range_iter = random.sample(range_no_id,
                                       self.params['poison_train_batch_size'])
            indices.extend(range_iter)

        ## poison dataset size 64 \times 200 (64: batch size, 200 batch)
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['poison_train_batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                           drop_last=True)

    def _get_train(self, indices):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                    batch_size = self.params["train_batch_size"],
                                    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices))
                                    # drop_last=True)

    def _get_test(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                    batch_size = self.params["test_batch_size"],
                                    shuffle=True)

    def _get_global_dataloader(self):
        indices = list()
        for batches in range(self.params["global_no_reuse"]):
            if len(self.global_data_indices) == self.params["global_data_batch_size"]:
                range_iter = self.global_data_indices
            else:
                range_iter = random.sample(self.global_data_indices, self.params["global_data_batch_size"])
            # range_iter = self.global_data_indices
            indices.extend(range_iter)

        return torch.utils.data.DataLoader(self.train_dataset,
                                       batch_size=self.params["global_data_train_batch_size"],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                       drop_last=True)

    def _get_sample(self):
        r'''
        sample limited ood data as open set noise
        '''
        ood_data = list()
        ood_data_label = list()
        sample_index = random.sample(range(len(self.ood_dataset)), self.params["ood_data_sample_lens"])
        for ind in sample_index:
            ood_data.append(self.ood_dataset[ind])
            assigned_label = random.randint(0,9)
            ood_data_label.append(assigned_label)
        return ood_data, ood_data_label

    # def _get_id_dataloader(self):
    #     samples_per_class = 20
    #     num_classes = self.params['class_num']
    #
    #
    #     class_indices = {i: [] for i in range(num_classes)}
    #     for idx, (_, label) in enumerate(self.test_dataset):
    #         class_indices[label].append(idx)
    #
    #     selected_indices = []
    #     for cls in range(num_classes):
    #         selected_indices.extend(np.random.choice(class_indices[cls], samples_per_class, replace=False))
    #
    #     balanced_subset = torch.utils.data.Subset(self.test_dataset, selected_indices)
    #
    #     subset_loader = torch.utils.data.DataLoader(balanced_subset, self.params["ood_data_batch_size"], shuffle=True,drop_last=True)
    #     return subset_loader

    def _get_id_dataloader(self, mode="dirichlet", alpha=0.4, total_samples=200, fixed_per_class=None):
        """
        mode: "dirichlet" 或 "fixed"
        alpha: Dirichlet 分布参数
        total_samples: 总样本数（仅 mode="dirichlet" 时用）
        fixed_per_class: 每类固定采样数量（仅 mode="fixed" 时用）
        """

        num_classes = self.params['class_num']

        # 收集每个类别的索引
        class_indices = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(self.test_dataset):
            class_indices[label].append(idx)

        selected_indices = []

        if mode == "fixed":
            if fixed_per_class is None:
                raise ValueError("mode='fixed' 时必须指定 fixed_per_class")
            for cls in range(num_classes):
                count = min(fixed_per_class, len(class_indices[cls]))
                selected_indices.extend(
                    np.random.choice(class_indices[cls], count, replace=False)
                )

        elif mode == "dirichlet":
            class_ratios = np.random.dirichlet([alpha] * num_classes)
            samples_per_class = (class_ratios * total_samples).astype(int)

            diff = total_samples - samples_per_class.sum()
            if diff > 0:
                for i in np.random.choice(num_classes, diff, replace=True):
                    samples_per_class[i] += 1
            elif diff < 0:
                for i in np.random.choice(
                        np.where(samples_per_class > 0)[0], -diff, replace=True
                ):
                    samples_per_class[i] -= 1

            for cls in range(num_classes):
                count = min(samples_per_class[cls], len(class_indices[cls]))
                selected_indices.extend(
                    np.random.choice(class_indices[cls], count, replace=False)
                )
        else:
            raise ValueError("mode 必须是 'dirichlet' 或 'fixed'")

        # === 统计每类样本数量 ===
        labels_selected = [self.test_dataset[idx][1] for idx in selected_indices]
        class_count = Counter(labels_selected)
        print("每类样本数分布：", dict(class_count))

        subset = torch.utils.data.Subset(self.test_dataset, selected_indices)
        subset_loader = torch.utils.data.DataLoader(
            subset,
            self.params["ood_data_batch_size"],
            shuffle=True,
            drop_last=True
        )
        return subset_loader


    def _get_ood_dataloader(self):
        r'''
        sample limited ood data as open set noise
        '''
        indices = random.sample(range(len(self.ood_dataset)), self.params["ood_data_sample_lens"])

        ood_dataloader =  torch.utils.data.DataLoader(self.ood_dataset,
                                           batch_size=self.params["ood_data_batch_size"],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                           drop_last=True)
        ood_datalist = list(ood_dataloader)
        ood_datalist_shape = self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"] * self.params["ood_data_batch_size"] 
        assigned_labels = np.array([i for i in range(self.params["class_num"])] * \
            (ood_datalist_shape//self.params["class_num"]) + [i for i in range(ood_datalist_shape%self.params["class_num"])])
        np.random.shuffle(assigned_labels)
        assigned_labels = assigned_labels.reshape(self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"], self.params["ood_data_batch_size"])
        for batch_id, batch in enumerate(ood_datalist):
            data, targets = batch
            if self.params["dataset"].upper()=="EMNIST":
                ood_datalist[batch_id][0] = ood_datalist[batch_id][0][:,0,:,:].unsqueeze(axis=1)
            if self.params["ood_data_source"] == "EMNIST":
                ood_datalist[batch_id][0] = ood_datalist[batch_id][0].repeat(1,3,1,1)

            for ind in range(len(targets)):
                targets[ind] = assigned_labels[batch_id][ind]
        ood_dataloader=iter(ood_datalist)
        return ood_dataloader
    def _get_ood_dataloader1(self):
        r'''
        sample limited ood data as open set noise
        '''
        indices = random.sample(range(len(self.ood_dataset1)), self.params["ood_data_sample_lens"])

        ood_dataloader =  torch.utils.data.DataLoader(self.ood_dataset1,
                                           batch_size=self.params["ood_data_batch_size"],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                           drop_last=True)
        ood_datalist = list(ood_dataloader)
        ood_datalist_shape = self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"] * self.params["ood_data_batch_size"]
        assigned_labels = np.array([i for i in range(self.params["class_num"])] * \
            (ood_datalist_shape//self.params["class_num"]) + [i for i in range(ood_datalist_shape%self.params["class_num"])])
        np.random.shuffle(assigned_labels)
        assigned_labels = assigned_labels.reshape(self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"], self.params["ood_data_batch_size"])
        for batch_id, batch in enumerate(ood_datalist):
            data, targets = batch
            if self.params["dataset"].upper()=="EMNIST":
                ood_datalist[batch_id][0] = ood_datalist[batch_id][0][:,0,:,:].unsqueeze(axis=1)
            if self.params["ood_data_source"] == "EMNIST":
                ood_datalist[batch_id][0] = ood_datalist[batch_id][0].repeat(1,3,1,1)

            for ind in range(len(targets)):
                targets[ind] = assigned_labels[batch_id][ind]
        ood_dataloader=iter(ood_datalist)
        return ood_dataloader

    def DataSet_distill_clean_data(self,model, dataloader, distill_data_name):
        model.eval()
        model.cuda()
        unloader = transforms.ToPILImage()
        list_clean_data_knowledge_distill = []
        for i, (input, target) in enumerate(dataloader):
            # print('target:', target[0])
            # sys.exit()
            # if distill_data_name=="cifar100":
            #     if target[0] in [13, 58, 81, 89]:
            #         # print(target[0])
            #         continue
            input, target = input.cuda(), target.cuda()
            # compute output
            with torch.no_grad():
                output = model(input)
            # print('Output size:', output.size())
            # print(output)
            for j in range(input.size(0)):  # 遍历批次中的每个样本

                input_i = input[j]  # 获取第j个样本的输入

                output_i = output[j]  # 获取第j个样本的输出

                # 转换成 PIL 图像
                input_i = unloader(input_i)

                list_clean_data_knowledge_distill.append((input_i, output_i))
        torch.save(list_clean_data_knowledge_distill, './dataset/distill_' + distill_data_name)





    def perform_attack(self, model):
        print("begin collect shadow datasets")
        cifar10_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        tiny_imagenet_transforms = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
        batch_size =self.params['poison_train_batch_size']
        distill_data_name=f'Compressed{self.params["shadow_datasets"]}for{self.params["dataset"]}'
        if self.params['shadow_datasets']=='Random':
            if self.params["dataset"].upper() == "TINY-IMAGENET":
                shadow_dataset=RandomImages(transform=tiny_imagenet_transforms, data_num=2000)
            elif self.params["dataset"].upper() == "CIFAR10":
                shadow_dataset = RandomImages(transform=cifar10_transforms, data_num=2000)
        train_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=batch_size)
        distill_data_path = './dataset/distill_' + distill_data_name
        if not os.path.exists(distill_data_path):
            print("开始蒸馏")
            self.DataSet_distill_clean_data(model,train_loader, distill_data_name)
        #使用全局模型数据蒸馏
        dataset = torch.load(distill_data_path)
        random.shuffle(dataset)
        data_num = len(dataset)
        images = []
        outputs = []
        for i in range(data_num):
            img = np.array(dataset[i][0]).flatten()
            output = np.array(dataset[i][1].cpu())
            img = img.reshape(1, -1)
            images.append(preprocessing.normalize(img, norm='l2').squeeze())
            output = output.reshape(1, -1)
            outputs.append(preprocessing.normalize(output, norm='l2').squeeze())
        images = np.array(images)
        outputs = np.array(outputs)
        batch_num = int(data_num / batch_size) + (data_num % batch_size != 0)
        data_compression = []
        com_ratio=0.5
        def select_img(images_batch, outputs_batch, batch_n):
            data_num = images_batch.shape[0]
            max_num = int(data_num * com_ratio)
            if max_num == 0:
                return
            n_selected = 0
            images_sim = np.dot(images_batch, images_batch.transpose())
            # print(images_sim)
            # sys.exit()
            outputs_sim = np.dot(outputs_batch, outputs_batch.transpose())
            co_sim = np.multiply(images_sim, outputs_sim)
            # print(co_sim)
            # sys.exit()

            index = random.randint(0, data_num - 1)
            # print(index)

            while n_selected < max_num:
                index = np.argmin(co_sim[index])
                data_compression.append(dataset[batch_n * batch_size + index])
                n_selected += 1
                co_sim[:, index] = 1

        compression_path='./dataset/compression_' + distill_data_name + '_' + str(com_ratio)
        if not os.path.exists(compression_path):
            for i in range(batch_num):
                images_batch = images[i * batch_size:min((i + 1) * batch_size, data_num)]
                outputs_batch = outputs[i * batch_size:min((i + 1) * batch_size, data_num)]
                select_img(images_batch, outputs_batch, i)
            torch.save(data_compression, './dataset/compression_' + distill_data_name + '_' + str(com_ratio))
        train_dataset = torch.load('./dataset/compression_' + distill_data_name + '_' + str(com_ratio))
        #遍历这个数据集，对数据集的一部分打上trigger
        poison_ratio=0.0
        images = []
        labels = []
        for i in range(len(train_dataset)):
            img = train_dataset[i][0]
            label = train_dataset[i][1]
            images.append(img)
            labels.append(label)

        train_set =TensorDatasetImg(images, labels,self.params['dataset'])

        print('darkfed ready')
        return torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)

    def save_tiny_dataset(self,root, id_dic, transform_train_Imagenet, transform_test_Imagenet):

        train_dataset = TrainTinyImageNet(root, id=id_dic, transform=transform_train_Imagenet)
        val_dataset = ValTinyImageNet(root, id=id_dic, transform=transform_test_Imagenet)


        torch.save(train_dataset, root+'train_dataset.pt')
        torch.save(val_dataset, root+'val_dataset.pt')




    def create_loader(self):
        if self.params["sample_dirichlet"]:
            indices_per_participant_malicious = self._sample_dirichlet_train_data(
                    self.params["no_of_total_participants"],
                    alpha = 1000)

            indices_per_participant = self._sample_dirichlet_train_data(
                    self.params["no_of_total_participants"],
                    alpha = self.params["dirichlet_alpha"])

            for i in range(self.params["no_of_adversaries"]):
                indices_per_participant[i] = indices_per_participant_malicious[i]

            self.train_data = [self._get_train(indices) for pos, indices in
                                  indices_per_participant.items()]
        # client_0_dataloader = None
        # target_client_id = 0
        #
        # # 1. 找到 0号客户端的 DataLoader
        # if isinstance(self.train_data, list) and len(self.train_data) > target_client_id:
        #     client_0_dataloader = self.train_data[target_client_id]
        # else:
        #     for client_id, dataloader in enumerate(self.train_data):
        #         if client_id == target_client_id:
        #             client_0_dataloader = dataloader
        #             break
        #     if client_0_dataloader is None:
        #         print(f"Error: Client {target_client_id} not found in self.train_data.")
        #         exit()
        #
        # # 2. 收集 0号客户端的所有标签
        # all_labels_client_0 = []
        # if client_0_dataloader:
        #     for _, labels in client_0_dataloader:
        #         if isinstance(labels, torch.Tensor):
        #             all_labels_client_0.extend(labels.cpu().tolist())
        #         else:
        #             all_labels_client_0.extend(labels)
        #
        # # 3. 统计标签分布
        # client_0_label_count = Counter(all_labels_client_0)
        #
        # # ************************************************
        # # 新增: 查找并打印数量最多的类别
        # # ************************************************
        # if client_0_label_count:  # 确保计数器不为空
        #     # most_common(1) 返回一个列表，包含一个元组 (元素, 计数)
        #     most_common_item = client_0_label_count.most_common(1)
        #     if most_common_item:  # 再次确认列表不是空的
        #         most_frequent_label = most_common_item[0][0]
        #         highest_count = most_common_item[0][1]
        #         print("-" * 30)  # 加个分隔符，让输出更清晰
        #         print(f"For Client {target_client_id}:")
        #         print(f"  The most frequent label is: Class {most_frequent_label}")
        #         print(f"  Number of samples for this class: {highest_count}")
        #         print("-" * 30)
        #     else:
        #         # 这种情况理论上在 client_0_label_count 非空时不应发生
        #         print(f"Could not determine the most frequent class for Client {target_client_id}.")
        # else:
        #     print(f"No labels found for Client {target_client_id}, cannot determine the most frequent class.")
        # # ************************************************
        # # 结束新增部分
        # # ************************************************
        #
        # # 4. 准备绘图数据 (如果需要绘图，保留这部分)
        # if client_0_label_count:
        #     sorted_labels = sorted(client_0_label_count.keys())
        #     counts = [client_0_label_count[label] for label in sorted_labels]
        #     label_names = [f"Class {label}" for label in sorted_labels]
        #
        #     # 5. 绘制柱状图 (如果需要绘图，保留这部分)
        #     fig, ax = plt.subplots(figsize=(8, 5))
        #     ax.bar(label_names, counts)
        #     ax.set_xlabel("Label Class")
        #     ax.set_ylabel("Number of Samples")
        #     ax.set_title(f"Label Distribution for Client {target_client_id}")
        #     plt.xticks(rotation=45, ha='right')
        #     plt.tight_layout()
        #     plt.show()
        #
        #     # 打印详细统计结果 (可选)
        #     print(f"\nFull label distribution for Client {target_client_id}:")
        #     for label, count in sorted(client_0_label_count.items()):
        #         print(f"  Class {label}: {count}")
        #
        # # 如果 client_0_label_count 为空，则不绘图也不打印详细信息
        # elif not client_0_label_count:  # 之前已经打印过 "No labels found..."
        #     pass  # 或者可以再加一条消息，说明因此无法绘图
        self.test_data = self._get_test()

        self.ood_data = self._get_ood_dataloader()
        # self.ood_data1=self._get_ood_dataloader1()
        if self.params["defense_method"].lower()=="ours":
            self.id_data=self._get_id_dataloader()
        self.poison_data = self._get_poison_train()
        self.edge_poison_train, self.edge_poison_test = self._load_edge_case()

        return True


class TensorDatasetImg(Dataset):
    def __init__(self, data_tensor, target_tensor,datasets):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.datasets=datasets
    def __getitem__(self, index):
        # img = copy.copy(self.data_tensor[index])        #print(type(img))
        img = self.data_tensor[index]
        if self.datasets == "tiny-imagenet":
            transform_train = transforms.Compose([
                # transforms.RandomCrop(64, padding=8),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        img = transform_train(img)
        label=self.target_tensor[index]
        img=img
        label=label
        #label=torch.argmax(label)
        return img, label
    def __len__(self):
        return len(self.data_tensor)


class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(os.path.join(root, "train", "*", "*", "*.JPEG"))
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split(os.sep)[-3]]  # Use os.sep to handle platform differences
        if self.transform:
            image = self.transform(image)
        return image, label


class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(os.path.join(root, "val", "images", "*.JPEG"))
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        with open(os.path.join(root, 'val', 'val_annotations.txt'), 'r') as file:
            for line in file:
                a = line.split('\t')
                img, cls_id = a[0], a[1]
                self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[os.path.basename(img_path)]  # Use os.path.basename for platform-independent filename extraction
        if self.transform:
            image = self.transform(image)
        return image, label