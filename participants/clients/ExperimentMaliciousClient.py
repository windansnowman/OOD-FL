import torch
import torch.nn as nn
from torchvision import transforms
from participants.clients.AbstractClient import AbstractClient
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import time
import math
import copy
logger = logging.getLogger("logger")
import models.cnn
import models.vgg
import models.resnet

from utils.utils import add_trigger

class ExperimentMaliciousClient(AbstractClient):
    def __init__(self, params, train_dataset, blend_pattern, open_set, 
                 edge_case_train, edge_case_test, open_set_label=None):
        super(ExperimentMaliciousClient, self).__init__(params)
        self.train_dataset = train_dataset
        self.sample_data, _ = self.train_dataset[1]
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test
        self.classifier_name = "linear"
        self.blend_pattern = blend_pattern
        self.open_set = open_set
        self.open_set_label = open_set_label
        self._create_check_model()
        self._loss_function()
        self.init_trigger()

    def init_trigger(self):
        trigger_size=5
        channel, height, width = self.sample_data.shape
        # loc_x = random.randint(0, height - trigger_size)
        # loc_y = random.randint(0, width - trigger_size)
        loc_x = 0
        loc_y = 0
        # self.trigger = torch.ones((1,channel,height,width), requires_grad=False, device='cuda') * 1
        self.trigger = (torch.rand((1,channel,height,width),requires_grad=False, device='cuda') - 0.5) * 2
        self.mask = torch.zeros_like(self.trigger)
        # self.mask[:, :, loc_x:loc_x+trigger_size, loc_y:loc_y +trigger_size] = 1
        self.mask = self.mask.cuda()
        self.trigger0 = copy.deepcopy(self.trigger)
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
        elif "CNN" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.cnn, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.cnn, self.params["model_type"])(num_classes=100)
        self.check_model = check_model.cuda()
        return True

    def soft_cross_entropy(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum()/input.shape[0]

    def combined_cross_entropy(self, input, target):
        loss = 1 - nn.functional.cosine_similarity(input, target, dim=1).item()
        # logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return loss/input.shape[0]

    def ceriterion_build(self, input, target, soft_label=False, reduction=None):
        if soft_label:
            loss = self.combined_cross_entropy(input, target)
        else:
            loss = nn.functional.cross_entropy(input, target, reduction=reduction)
        
        return loss

    def _loss_function(self):
        # self.ceriterion = self.ceriterion_build
        self.ceriterion = nn.functional.cross_entropy
        return True

    # def _loss_function(self):
    #     self.ceriterion = nn.functional.cross_entropy
    #
    #     return True

    def _optimizer(self, round, adaptive):
        if adaptive:
            lr = self.params["adaptive_attack_lr"]
        else:
            lr = self.params["poisoned_lr"]
        logger.info(f"malicious lr:{lr}")
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr,
                                    momentum=self.params["poisoned_momentum"], weight_decay=self.params["poisoned_weight_decay"])  
        return True

    def _scheduler(self, adaptive):
        if adaptive:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['adaptive_malicious_milestones'],
                                                 gamma=self.params['adaptive_malicious_lr_gamma'])
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['malicious_milestones'],
                                                 gamma=self.params['malicious_lr_gamma'])
            
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

    def _projection(self, target_params_variables):
        model_norm = self._model_dist_norm(self.local_model, target_params_variables)
        if self.params["show_train_log"]:
            logger.info(f"model dist is :{model_norm}")

        if model_norm > self.params["poisoned_projection_norm"] and self.params["poisoned_is_projection_grad"]:
            norm_scale = self.params["poisoned_projection_norm"] / model_norm
            for name, param in self.local_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)

        return True

    def batch_label_distrib(self, targets):
        distrib_dict=dict()
        no_class = 100 if self.params["dataset"].upper()=="CIFAR100" else 10
        for label in range(no_class):
            distrib_dict[label] = 0
        sum_no = 0
        
        for label in targets:
            label = label.item()
            distrib_dict[label] += 1
            sum_no+=1

        percentage_dict=dict()
        for key,value in distrib_dict.items():
            percentage_dict[key] = round(value/sum_no, 2)

        return distrib_dict, percentage_dict, sum_no

    def _grad_mask_cv(self, model, clean_data, ratio=None):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        for internal_round in range(10):
            for inputs, labels in clean_data:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = nn.functional.cross_entropy(output, labels)
                loss.backward(retain_graph=True)
        mask_grad_list = []

        if self.params['malicious_aggregate_all_layer'] == 1:
            grad_list = []
            grad_abs_sum_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            grad_list = torch.cat(grad_list).cuda()
            if not isinstance(ratio, list):
                _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))
                mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
                mask_flat_all_layer[indices] = 1.0

            else:
                left_ratio = ratio[0]
                right_ratio = ratio[1]
                _, left_indices = torch.topk(grad_list, int(len(grad_list) * left_ratio))
                _, right_indices = torch.topk(grad_list, int(len(grad_list) * right_ratio))
                mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
                mask_flat_all_layer[right_indices] = 1.0
                mask_flat_all_layer[left_indices] = 0.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    count += gradients_length
                    percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0
                    percentage_mask_list.append(percentage_mask1)
                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list))
                    k_layer += 1

        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda()) / float(
                        len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1 * gradients, int(gradients_length * 1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1 * gradients, int(gradients_length * ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                    percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0
                    percentage_mask_list.append(percentage_mask1)
                    k_layer += 1

        model.zero_grad()
        return mask_grad_list



    def local_training(self, train_data, test_data, target_params_variables, is_log_train, poisoned_pattern_choose=None, round=None, model_id=None,former_model=None):
        total_loss = 0
        data_iterator = train_data
        self.search_trigger(self.local_model,data_iterator)
        self._loss_function()

        if self.params["adaptive_attack"]:
            self._optimizer(round, adaptive=True)
            self._scheduler(adaptive=True)
        else:
            self._optimizer(round, adaptive=False)
            self._scheduler(adaptive=False)
        if self.params["adaptive_attack"]:
            retrain_no_times = self.params["adaptive_attack_round"] + self.params["poisoned_retrain_no_times"]
        else:
            retrain_no_times = self.params["poisoned_retrain_no_times"]

        for internal_round in range(retrain_no_times):
            logger.info(f"Malicious training: plr:{self.optimizer.state_dict()['param_groups'][0]['lr']}")
            for batch_id, batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                mask_grad_list = self._grad_mask_cv(model=self.local_model, clean_data=train_data,
                                                    ratio=self.params["malicious_neurotoxin_ratio"])
                if self.params["adaptive_attack"] and internal_round < self.params["adaptive_attack_round"]:
                    batch = copy.deepcopy(batch)
                else:
                    batch = self._poisoned_batch_injection(batch,evaluation=False)

                data, targets = batch

                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = self.local_model(data)
                # loss = self.ceriterion(output, targets)
                class_loss = self.ceriterion(output, targets)
                distance_loss = self._model_dist_norm_var(self.local_model, target_params_variables)
                loss = class_loss
                loss.backward()
                self._apply_grad_mask(self.local_model, mask_grad_list)
                self.optimizer.step()

                self._projection(target_params_variables)
                total_loss += loss.data

                # if batch_id % 10 == 0 and is_log_train:
                # if batch_id % 2 == 0 and is_log_train:
                # if batch_id == len(data_iterator)-1 and is_log_train:
                # if batch_id == len(data_iterator)-1 and internal_round == self.params["benign_retrain_no_times"]-1 and is_log_train:
                if is_log_train:

                    loss, acc = self._local_test_sub(test_data, test_poisoned=False, model=self.local_model)
                    logger.info(f"round:{internal_round} | benign acc:{acc}, benign loss:{loss}")

                    loss_p, acc_p = self._local_test_sub(test_data, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose, model=self.local_model)
                    logger.info(f"round:{internal_round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")
                    
                    wm_data=copy.deepcopy(self.open_set)
                    loss_w, acc_w, label_acc_w = self._local_watermarking_test_sub(wm_data, model=self.local_model)
                    logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target class wm acc:{label_acc_w}")

                    logger.info(f" ")

            if self.params["adaptive_attack"] and \
                internal_round==self.params["adaptive_attack_round"]-1:
                self._optimizer(round, adaptive=False)
                self._scheduler(adaptive=False)
            else:
                self.scheduler.step()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        image = (self.trigger.detach().cpu()[0] * std + mean).detach().cpu()
        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        image.save(f"output{round}.png")
            # 显示图片
        plt.imshow(image)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        image = (data.detach().cpu()[0] * std + mean).detach().cpu()
        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        image.save(f"output{round}.png")
        # 显示图片
        plt.imshow(image)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        image = ((data.detach().cpu()[0]-self.trigger.detach().cpu()[0]) * std + mean).detach().cpu()
        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        image.save(f"output{round}.png")
        # 显示图片
        plt.imshow(image)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
    # def get_adv_model(self, model, dl, trigger, mask):
    #     adv_model = copy.deepcopy(model)
    #     adv_model.train()
    #     ce_loss = torch.nn.CrossEntropyLoss()
    #     adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    #     for _ in range(2):
    #         for inputs, labels in dl:
    #             inputs, labels = inputs.cuda(), labels.cuda()
    #             inputs = trigger * mask + (1 - mask) * inputs
    #             outputs = adv_model(inputs)
    #             loss = ce_loss(outputs, labels)
    #             adv_opt.zero_grad()
    #             loss.backward()
    #             adv_opt.step()
    #
    #     sim_sum = 0.
    #     sim_count = 0.
    #     cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    #     for name in dict(adv_model.named_parameters()):
    #         if 'conv' in name or 'features' in name:
    #             sim_count += 1
    #             sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1), \
    #                                 dict(model.named_parameters())[name].grad.reshape(-1))
    #     return adv_model, sim_sum / sim_count
    #
    # def search_trigger(self, model, dl):
    #     ce_loss = torch.nn.CrossEntropyLoss()
    #     alpha = 0.02
    #
    #     K = 200
    #     t = self.trigger.clone()
    #     m = self.mask.clone()
    #     normal_grad = 0.
    #     count = 0
    #     adv_model=copy.deepcopy(model)
    #     adv_w=1
    #     for iter in range(K):
    #         if iter!=0:
    #             adv_model, adv_w = self.get_adv_model(model, dl, t, m)
    #         for inputs, labels in dl:
    #             count += 1
    #             t.requires_grad_()
    #             inputs, labels = inputs.cuda(), labels.cuda()
    #             inputs = t * m + (1 - m) * inputs
    #             labels[:] = self.params["poison_label_swap"]
    #             outputs = model(inputs)
    #             loss = ce_loss(outputs, labels)
    #             outputs = adv_model(inputs)
    #             nm_loss = ce_loss(outputs, labels)
    #             if loss == None:
    #                     loss = 0.1 * adv_w * nm_loss
    #             else:
    #                     loss += 0.1 * adv_w * nm_loss
    #             if loss != None:
    #                 loss.backward()
    #                 normal_grad += t.grad.sum()
    #                 new_t = t - alpha * t.grad.sign()
    #                 t = new_t.detach_()
    #                 t = torch.clamp(t, min=-2.5, max=2.5)
    #                 t.requires_grad_()
    #     t = t.detach()
    #     self.trigger = t
    #     self.mask = m
    def generate_discriminator_dataloader(self, model, train_loader):
        '''
        discriminator trainset, target class is 0, target class is 1
        :param model:
        :param train_loader:
        :param trigger_:
        :param mask_:
        :param client_id:
        '''

        class_num = self.params["class_num"]
        samples_per_class = {i: torch.tensor([], device='cuda:0') for i in range(class_num)}
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        label_list = [0 for _ in range(class_num)]
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            for class_ind in range(class_num):
                indices = labels == class_ind
                label_list[class_ind] += sum(indices)
                samples_per_class[class_ind] = torch.cat((samples_per_class[class_ind], inputs[indices]), dim=0)

        target_class = self.params["poison_label_swap"]
        for i in range(class_num):
            sample = samples_per_class[i]
            if len(sample) == 0:
                continue
            outputs = model(sample)
            tmp_label = torch.ones(len(outputs), dtype=torch.long, device='cuda:0') * i
            loss_sort_by_samples = criterion(outputs, tmp_label)
            samples_selected_len = self.params["discriminator_train_samples_pre_class"] if len(outputs) > self.params[
                "discriminator_train_samples_pre_class"] else len(outputs)
            if i == target_class:
                samples_selected_len = len(outputs)
            _, indices = torch.topk(loss_sort_by_samples, samples_selected_len,
                                    largest=False)
            representative_samples = sample[indices]
            samples_per_class[i] = representative_samples
        samples_discriminator_dataloader = torch.tensor([], device='cuda:0')
        labels_discriminator_dataloader = torch.tensor([], dtype=torch.long, device='cuda:0')
        for i in range(class_num):
            if i == target_class:
                continue
            samples = samples_per_class[i]
            labels = torch.ones(len(samples), dtype=torch.long, device='cuda:0')
            poisoned_sample, _ = self.poisoned_batch_injection((backdoor_inputs,backdoor_targets),t=t,m=m,evaluation=False)
            samples_discriminator_dataloader = torch.cat((samples_discriminator_dataloader, poisoned_sample), dim=0)
            labels_discriminator_dataloader = torch.cat((labels_discriminator_dataloader, labels), dim=0)

        samples_discriminator_dataloader = torch.cat(
            (samples_discriminator_dataloader, samples_per_class[target_class]), dim=0)
        labels_discriminator_dataloader = torch.cat((labels_discriminator_dataloader,
                                                     torch.zeros(len(samples_per_class[target_class]), dtype=torch.long,
                                                                 device='cuda:0')), dim=0)
        discriminator_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(samples_discriminator_dataloader, labels_discriminator_dataloader),
            batch_size=self.params["discriminator_batch_size"], shuffle=True)

        return discriminator_dataloader

    def get_discriminator(self, model, discriminator_dataloader):
        discriminator_ = copy.deepcopy(model)
        if "resnet" in self.params["model_type"].lower():
            discriminator_.linear = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.linear.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "vgg" in self.params["model_type"].lower():
            discriminator_.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.classifier.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "mobilenet" in self.params["model_type"].lower():
            discriminator_.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.classifier[1].in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )

        for name, param in discriminator_.named_parameters():
            if self.classifier_name not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        discriminator_optimizer = torch.optim.SGD(discriminator_.parameters(), lr=self.params["discriminator_lr"],
                                                  momentum=self.params['discriminator_momentum'],
                                                  weight_decay=self.params['discriminator_weight_decay'])

        discriminator_criterion = nn.CrossEntropyLoss().cuda()

        discriminator_ = discriminator_.cuda()

        for iter in range(self.params["discriminator_train_no_times"]):
            total_loss = 0.
            for batch in discriminator_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = discriminator_(inputs)
                loss = discriminator_criterion(outputs, labels)
                discriminator_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                total_loss += loss.item()
                discriminator_optimizer.step()
        discriminator_.eval()

        return discriminator_

    def search_trigger(self, model, train_loader):
        '''
        optimize trigger

        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''
        model.eval()
        local_train_loader = copy.deepcopy(train_loader)
        ce_loss = nn.functional.cross_entropy
        cos_loss = nn.CosineSimilarity(dim=1, eps=1e-08)

        feature_extractor = copy.deepcopy(model)
        feature_extractor.linear = torch.nn.Sequential()

        t = self.trigger
        m = self.mask
        for iters in range(self.params["trigger_search_no_times"]):
            dataloader_discriminator = self.generate_discriminator_dataloader(model, local_train_loader)
            total_loss = 0.
            trigger_optim = torch.optim.Adam([t], lr=self.params["trigger_lr"], weight_decay=5e-4)
            counter = 0
            loss_adv = 0.
            loss_acc = 0.
            model_discriminator = self.get_discriminator(model, dataloader_discriminator)

            for inputs, targets in train_loader:  # 在训练集上更新样本
                t.requires_grad_()
                inputs, targets = inputs.cuda(), targets.cuda()
                batch_clean_indices = targets == self.params["poison_label_swap"]
                counter += 1

                batch_backdoor_indices = ~batch_clean_indices
                backdoor_inputs = inputs[batch_backdoor_indices]
                backdoor_targets = targets[batch_backdoor_indices]
                # backdoor_inputs = t * m + (1 - m) * backdoor_inputs
                # backdoor_targets[:] = self.params["poison_label_swap"]
                backdoor_inputs, backdoor_targets=self.poisoned_batch_injection((backdoor_inputs,backdoor_targets),t=t,m=m,evaluation=False)
                # backdoor_inputs, backdoor_targets = self._poisoned_batch_injection((backdoor_inputs, backdoor_targets),evaluation=False)

                backdoor_inputs = backdoor_inputs.cuda()

                # TODO 1 -> to ID,
                backdoor_pred_disc = model_discriminator(backdoor_inputs)
                loss_discriminator = ce_loss(backdoor_pred_disc,
                                             torch.zeros(len(backdoor_pred_disc),
                                                         device='cuda:0').long())
                backdoor_pred = model(backdoor_inputs)
                # TODO 2 -> enhancement
                loss_asr = ce_loss(backdoor_pred, backdoor_targets)
                loss_sim = cos_loss(backdoor_pred, model(inputs[batch_backdoor_indices])).mean()

                loss = 0.
                loss += loss_discriminator
                loss += loss_asr
                loss += loss_sim
                total_loss += loss.item()
                if loss != None and loss.item() != 0.:
                    trigger_optim.zero_grad()
                    loss.backward(retain_graph=True)
                    new_t = t - t.grad.sign() * self.params["trigger_lr"]
                    t = new_t.detach()
                    t = torch.clamp(t, min=-0.15, max=0.15)
                    t.requires_grad_()
            model.train()
            self.trigger = t



    # def Adv_trigger(self, model, dl):
    #     ce_loss = torch.nn.CrossEntropyLoss()
    #     model.eval()
    #     alpha = 0.005
    #     K = 50
    #     t = self.trigger.clone()
    #     m = self.mask.clone()
    #     normal_grad = 0.
    #     count = 0
    #     for iter in range(K):
    #         for inputs, labels in dl:
    #             count += 1
    #             t.requires_grad_()
    #             inputs, labels = inputs.cuda(), labels.cuda()
    #             inputs = t * m + (1 - m) * inputs
    #             labels[:] =self.params["poison_label_swap"]
    #             outputs = model(inputs)
    #             loss_adv = ce_loss(outputs, labels)
    #             loss=loss_adv
    #             if loss != None:
    #                 loss.backward()
    #                 normal_grad += t.grad.sum()
    #                 new_t = t - alpha * t.grad.sign()
    #                 t = new_t.detach_()
    #                 t = torch.clamp(t, min =-2.5, max =2.5)
    #                 t.requires_grad_()
    #     t = t.detach()
    #     self.trigger = t
    #     self.mask = m
    #     model.train()

    def _poisoned_batch_injection(self, batch, evaluation=False):
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
                if self.params['poisoned_pattern_choose'] == 10:
                    # poisoned_batch[0][pos] = poisoned_batch[0][pos]*(1-self.params['blend_alpha']) + self.trigger.to(poisoned_batch[0][pos].device)*self.params['blend_alpha']
                    poisoned_batch[0][pos] = poisoned_batch[0][pos] + self.trigger.to(poisoned_batch[0][pos].device)
                elif self.params['poisoned_pattern_choose'] == 1:
                    poisoned_batch[0][pos] = poisoned_batch[0][pos] * (1 - self.mask.to(poisoned_batch[0][pos].device)) + self.trigger.to(
                        poisoned_batch[0][pos].device) * self.mask.to(poisoned_batch[0][pos].device)
                poisoned_batch[1][pos] = self.params["poison_label_swap"]
        return poisoned_batch

    def poisoned_batch_injection(self, batch, t,m,evaluation=False):
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
                if self.params['poisoned_pattern_choose'] == 10:
                    # poisoned_batch[0][pos] = poisoned_batch[0][pos] * (
                    #             1 - self.params['blend_alpha']) + t.to(poisoned_batch[0][pos].device) * \
                    #                          self.params['blend_alpha']
                    poisoned_batch[0][pos]=poisoned_batch[0][pos]+ t.to(poisoned_batch[0][pos].device)
                elif self.params['poisoned_pattern_choose'] == 1:
                    poisoned_batch[0][pos] = poisoned_batch[0][pos]*(1-m.to(poisoned_batch[0][pos].device)) + t.to(poisoned_batch[0][pos].device)*m.to(poisoned_batch[0][pos].device)
                poisoned_batch[1][pos] = self.params["poison_label_swap"]

        return poisoned_batch

    def _local_watermarking_test_sub(self, test_data, model=None):
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        dataset_size = 0
        correct = 0
        wm_label_correct = 0
        wm_label_sum = 0
        data_iterator = copy.deepcopy(test_data)
        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):

                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = model(data)
                total_loss += self.ceriterion(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]

                if batch_id==0 and self.params["show_train_log"]:
                    logger.info(f"watermarking pred: {pred}")
                    logger.info(f"watermarking targets: {targets}")

                poisoned_label = self.params["poison_label_swap"]
                wm_label_targets = torch.ones_like(targets) * poisoned_label
                wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))
                wm_label_sum += wm_label_index.cpu().sum().item()
                wm_label_correct += pred.eq(targets.data.view_as(pred))[wm_label_index.bool()].cpu().sum().item()

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                dataset_size += len(targets)
            
        watermark_acc = 100.0 *(float(correct) / float(dataset_size))
        wm_label_acc = 100.0 *(float(wm_label_correct) / float(wm_label_sum))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, watermark_acc, wm_label_acc)

    def _local_test_sub(self, test_data, model=None, test_poisoned=False, poisoned_pattern_choose=None):

        if model == None:
            model = self.local_model

        model.eval()
        total_loss = 0
        correct = 0
        dataset_size = len(test_data.dataset)

        data_iterator = test_data
        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):
                if test_poisoned:
                    poisoned_batch = self._poisoned_batch_injection(batch,evaluation=True)
                else:
                    poisoned_batch = copy.deepcopy(batch)

                data, targets = poisoned_batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = model(data)
                total_loss += self.ceriterion(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]

                clean_batch = copy.deepcopy(batch)
                _,clean_targets = clean_batch
                clean_targets = clean_targets.cuda().detach().requires_grad_(False)

                # if batch_id==0 and test_watermarking:
                #     logger.info(f"watermarking preds are:{pred}")
                #     logger.info(f"watermarking target labels are:{targets}")

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, acc)

    def local_test(self, model_id, test_data, round, poisoned_pattern_choose=None, model=None):

        loss, acc = self._local_test_sub(test_data, test_poisoned=False, model=model)
        logger.info(f"model:{model_id}, round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._local_test_sub(test_data, test_poisoned=True,
                                             poisoned_pattern_choose=poisoned_pattern_choose, model=model)
        logger.info(f"model:{model_id}, round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return True

    def global_test(self, model, test_data, round, poisoned_pattern_choose=None):

        loss, acc = self._local_test_sub(test_data,model,test_poisoned=False)
        logger.info(f"model:global, round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._local_test_sub(test_data,model,test_poisoned=True,
                                             poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"model:global, round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return acc, acc_p
