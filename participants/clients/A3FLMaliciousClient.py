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

class A3FLMaliciousClient(AbstractClient):
    def __init__(self, params, train_dataset, blend_pattern, open_set, 
                 edge_case_train, edge_case_test, open_set_label=None):
        super(A3FLMaliciousClient, self).__init__(params)
        self.train_dataset = train_dataset
        self.sample_data, _ = self.train_dataset[1]
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

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
        self.trigger = torch.ones((1,channel,height,width), requires_grad=False, device='cuda') * 0.5
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, loc_x:loc_x+trigger_size, loc_y:loc_y +trigger_size] = 1
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

                if self.params["adaptive_attack"] and internal_round < self.params["adaptive_attack_round"]:
                    batch = copy.deepcopy(batch)
                else:
                    batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=False, model_id=model_id)

                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = self.local_model(data)
                # loss = self.ceriterion(output, targets)
                class_loss = self.ceriterion(output, targets)
                distance_loss = self._model_dist_norm_var(self.local_model, target_params_variables)
                loss = class_loss
                loss.backward()
                # self._apply_grad_mask(self.local_model, mask_grad_list)
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

    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(2):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name or 'features' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1), \
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum / sim_count

    def search_trigger(self, model, dl):
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = 0.02

        K = 50
        t = self.trigger.clone()
        m = self.mask.clone()
        normal_grad = 0.
        count = 0
        adv_model=copy.deepcopy(model)
        adv_w=1
        for iter in range(K):
            if iter!=0:
                adv_model, adv_w = self.get_adv_model(model, dl, t, m)
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t * m+ (1 - m) * inputs
                labels[:] = self.params["poison_label_swap"]
                outputs = model(inputs)
                loss = ce_loss(outputs, labels)
                outputs = adv_model(inputs)
                nm_loss = ce_loss(outputs, labels)
                if loss == None:
                        loss = 0.1 * adv_w * nm_loss
                else:
                        loss += 0.1 * adv_w * nm_loss
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2.5, max=2.5)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        model.train()


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
                if evaluation:
                    poisoned_batch[0][pos] = poisoned_batch[0][pos] +(self.trigger.detach().cpu() -poisoned_batch[0][pos])* self.mask.detach().cpu()
                else:
                    poisoned_batch[0][pos] = poisoned_batch[0][pos] +(self.trigger.detach().cpu() -poisoned_batch[0][pos])* self.mask.detach().cpu()*0.3

                poisoned_batch[1][pos] = self.params["poison_label_swap"]
            # if evaluation:
            #     poisoned_batch[0][pos] = poisoned_batch[0][pos] +(self.trigger.detach().cpu() -poisoned_batch[0][pos])* self.mask.detach().cpu()
            #     poisoned_batch[1][pos] = self.params["poison_label_swap"]
            # else:
            #     label=poisoned_batch[1][pos]
            #     if label==self.params["poison_label_swap"]:
            #         poisoned_batch[0][pos] = poisoned_batch[0][pos] +(self.trigger.detach().cpu() -poisoned_batch[0][pos])* self.mask.detach().cpu()*0.3


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
                    poisoned_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
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
