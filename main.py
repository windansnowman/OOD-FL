#This code is based on https://github.com/ybdai7/Backdoor-indicator-defense, we added some new defense and attack methods.
import torch
import os
import numpy as np
import argparse
import yaml
import random
import datetime
import logging
from participants.servers.IndicatorServer import IndicatorServer
from participants.servers.FlameServer import FlameServer
from participants.servers.NodefenseServer import NodefenseServer
from participants.servers.FoolsgoldServer import FoolsgoldServer
from participants.servers.DeepsightServer import DeepsightServer
from participants.servers.RflbatServer import RflbatServer
from participants.servers.MultikrumServer import MultikrumServer
from participants.servers.OursServer import OursServer
from participants.servers.FedRecover import FedRecover
from participants.servers.AlignInsServer import AlignInsServer
from participants.servers.FedDMCServer import FedDMCServer
from participants.servers.MultiMetricsServer import MultiMetricsServer
from participants.clients.FedProxBenignClient import FedProxBenignClient
from participants.servers.DatasetDisstiallionServer import DatasetDisstiallionServer
from participants.clients.MaliciousClient import MaliciousClient
from participants.clients.DarkFedMaliciousClient import DarkFedMaliciousClient
from participants.clients.ChameleonMaliciousClient import ChameleonMaliciousClient
from participants.clients.A3FLMaliciousClient import A3FLMaliciousClient
from participants.clients.PFedBAMaliciousClient import PFedBAMaliciousClient
from participants.clients.MirageMaliciousClient import MirageMaliciousClient
from participants.clients.WaNetClient import WaNetClient
from participants.clients.AdvBlendClient import AdvBlendClient
# from participants.clients.OursMaliciousClient import OursMaliciousClient
from participants.clients.MutilLabelMaliciousClient import MlabelMaliciousClient
from dataloader.WMFLDataloader import WMFLDataloader
from utils.utils import save_model
from utils.utils import plot_poisoned_acc
import copy

logger = logging.getLogger("logger")
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def update_the_Ss(state_dict1, state_dict2, alpha, global_model_state_dict):
    s1_new = {key: alpha * global_model_state_dict[key] + (1 - alpha) * state_dict1[key] for key in state_dict1.keys()}
    s2_new = {key: alpha * s1_new[key] + (1 - alpha) * state_dict2[key] for key in state_dict2.keys()}
    return s1_new, s2_new


def predict_the_global_model(state_dict1, state_dict2, alpha):
    # s1:state_dict()
    sum_state_dict = {key: ((2 - alpha) / 1 - alpha) * state_dict1[key] - (1 / (1 - alpha)) * state_dict2[key] for key
                      in state_dict1.keys()}

    return sum_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="utils/yamls")
    parser.add_argument("--GPU_id", default="0", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--class_num", default=10, type=int)
    parser.add_argument("--model_type", default="ResNet18", type=str,
                        help='tiny-imagenet is 64*64 so if tiny-imagenet use ResNet18tiny instead of ResNet18')
    parser.add_argument("--agg_method", default="FedProx", type=str)
    parser.add_argument("--benign_lr", default=0.1, type=float)
    parser.add_argument("--malicious_train_algo", default="vanilla", type=str,
                        help='you can use vanilla,PGD,Neurotoxin,Chameleon,DarkFed,A3FL,PFedBA,DBA.Blend,Edge you can change the type in .yaml.DarkFed really needs good parameter tuning')
    parser.add_argument("--poisoned_lr", default=0.07, type=float, help='400 : 0.1  800: 0.07  1200: 0.05')
    parser.add_argument("--defense_method", default="ours", type=str)
    parser.add_argument("--dirichlet_alpha", default=0.2
                        , type=float)
    parser.add_argument("--start_round", default=800, type=int)
    parser.add_argument("--end_round", default=820, type=int)
    parser.add_argument("--poisoned_start_round", default=800, type=int)
    parser.add_argument("--poisoned_end_round", default=900, type=int)
    parser.add_argument("--global_watermarking_start_round", default=800, type=int)
    parser.add_argument("--global_watermarking_end_round", default=1701, type=int)
    parser.add_argument("--replace_original_bn", default=True, type=bool)
    parser.add_argument("--no_of_adversaries", default=1, type=int)
    parser.add_argument("--ood_data_source", default='300KRANDOM', type=str,
                        help='You can also use images you find on the Internet or other ood datasets')
    parser.add_argument("--ood_data_sample_lens", default=800, type=int)
    parser.add_argument("--detection_alpha", default=0.8, type=float)
    parser.add_argument("--Adaptive_prune", default=False, type=bool)
    args = parser.parse_args()
    args.params = args.params + f"/{args.defense_method.lower()}/params_{args.malicious_train_algo.lower()}_{args.defense_method}.yaml"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    with open(f"./{args.params}", "r") as f:
        params_loaded = yaml.safe_load(f)
    params_loaded.update(vars(args))
    # First you have to set the start round to 1 to train a clean model before you can use resumed_model to import the trained global model. Also pay attention to the naming of the folder.
    params_loaded[
        'resumed_model'] = f'{args.dataset}_{args.model_type}_{args.dirichlet_alpha}/saved_model_global_model_{args.start_round}.pt.tar'
    if args.start_round == 1:
        params_loaded['resumed_model'] = False
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    dataloader = WMFLDataloader(params=params_loaded)
    # generate blend pattern
    sample_data, _ = dataloader.train_dataset[1]
    channel, height, width = sample_data.shape
    blend_pattern = (torch.rand(sample_data.shape) - 0.5) * 2

    if dataloader.params["defense_method"].lower() == "nodefense":
        # server = NodefenseServer(params=params_loaded, current_time=current_time,
        #                          train_dataset=dataloader.train_dataset,
        #                          blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
        #                          edge_case_test=dataloader.edge_poison_test)
        server = DatasetDisstiallionServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)
    elif dataloader.params["defense_method"].lower() == "indicator":
        server = IndicatorServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 open_set=dataloader.ood_data, blend_pattern=blend_pattern,
                                 edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "flame":
        server = FlameServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset,
                             blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                             edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "deepsight":
        server = DeepsightServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "foolsgold":
        server = FoolsgoldServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "rflbat":
        server = RflbatServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset,
                              blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                              edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "multikrum":
        server = MultikrumServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "ours":
        server = OursServer(params=params_loaded, current_time=current_time,
                            train_dataset=dataloader.train_dataset,
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test, id_data=dataloader.id_data,
                            open_set=dataloader.ood_data, test_dataset=dataloader.test_data)


    elif dataloader.params["defense_method"].lower() == "fedrecover":
        server = FedRecover(params=params_loaded, current_time=current_time,
                            train_dataset=dataloader.train_dataset,
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)


    elif dataloader.params["defense_method"].lower() == "alignins":
        server = AlignInsServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "feddmc":
        server = FedDMCServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower() == "multimetrics":
        server = MultiMetricsServer(params=params_loaded, current_time=current_time,
                                 train_dataset=dataloader.train_dataset,
                                 blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                 edge_case_test=dataloader.edge_poison_test)

    if server.params["agg_method"] == "FedProx":
        benign_client = FedProxBenignClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                            blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                            edge_case_train=dataloader.edge_poison_train,
                                            edge_case_test=dataloader.edge_poison_test)

    if server.params["malicious_train_algo"].upper() == "CHAMELEON":
        malicious_client = ChameleonMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                                    blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                                    edge_case_train=dataloader.edge_poison_train,
                                                    edge_case_test=dataloader.edge_poison_test)

    elif server.params["malicious_train_algo"].upper() == "DARKFED":
        shadow_data = dataloader.perform_attack(server.global_model)
        malicious_client = DarkFedMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                                  blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                                  edge_case_train=dataloader.edge_poison_train,
                                                  edge_case_test=dataloader.edge_poison_test,
                                                  shadow_datasets=shadow_data)

    elif server.params["malicious_train_algo"].upper() == "A3FL":
        malicious_client = A3FLMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                               blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                               edge_case_train=dataloader.edge_poison_train,
                                               edge_case_test=dataloader.edge_poison_test)

    elif server.params["malicious_train_algo"].upper() == "PFEDBA":
        malicious_client = PFedBAMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                                 blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                                 edge_case_train=dataloader.edge_poison_train,
                                                 edge_case_test=dataloader.edge_poison_test)

    elif server.params["malicious_train_algo"].upper() == "MIRAGE":
        malicious_client =MirageMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                               blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                               edge_case_train=dataloader.edge_poison_train,
                                               edge_case_test=dataloader.edge_poison_test)
    elif server.params["malicious_train_algo"].upper() == "WANET":
        malicious_client = WaNetClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                           blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                           edge_case_train=dataloader.edge_poison_train,
                                           edge_case_test=dataloader.edge_poison_test)
    elif server.params["malicious_train_algo"].upper() == "ADVBLEND":
        malicious_client = AdvBlendClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                           blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                           edge_case_train=dataloader.edge_poison_train,
                                           edge_case_test=dataloader.edge_poison_test)
    # elif server.params["malicious_train_algo"].upper() == "MUTILLABEL":
    #     malicious_client = MutilLableMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
    #                                              blend_pattern=blend_pattern, open_set=dataloader.ood_data,
    #                                              edge_case_train=dataloader.edge_poison_train,
    #                                              edge_case_test=dataloader.edge_poison_test)

    else:
        malicious_client = MaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
                                           blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                           edge_case_train=dataloader.edge_poison_train,
                                           edge_case_test=dataloader.edge_poison_test)
    # else:
    #     malicious_client = MlabelMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset,
    #                                                  blend_pattern=blend_pattern, open_set=dataloader.ood_data,
    #                                                  edge_case_train=dataloader.edge_poison_train,
    #                                                  edge_case_test=dataloader.edge_poison_test)

    acc_list = list()
    acc_p_list = list()
    former_model = None
    current_model = copy.deepcopy(server.global_model)
    s1 = current_model.state_dict()
    s2 = None
    for round in range(server.params["start_round"], server.params["end_round"]):

        server.pre_process(test_data=dataloader.test_data,
                           round=round
                           )
        if round == server.params["start_round"] + 1:
            s2 = server.global_model.state_dict()
        if round > server.params["start_round"] + 1:
            s1, s2 = update_the_Ss(s1, s2, 0.8, server.global_model.state_dict())
        predicted_model = copy.deepcopy(server.global_model)
        if round != server.params["start_round"]:
            new_state_dict = predict_the_global_model(s1, s2, alpha=0.8)
            predicted_model.load_state_dict(new_state_dict)
        if server.params["defense_method"].lower() != "flame":
            weight_accumulator, aggregated_model_id \
                = server.broadcast_upload(
                round=round,
                local_benign_client=benign_client,
                local_malicious_client=malicious_client,
                train_dataloader=dataloader.train_data,
                poison_train_dataloader=dataloader.poison_data,
                test_dataloader=dataloader.test_data, former_model=predicted_model
            )

            server.aggregation(weight_accumulator=weight_accumulator, aggregated_model_id=aggregated_model_id,
                               round=round)
        else:
            weight_accumulator, aggregated_model_id, clip_value \
                = server.broadcast_upload(
                round=round,
                local_benign_client=benign_client,
                local_malicious_client=malicious_client,
                train_dataloader=dataloader.train_data,
                poison_train_dataloader=dataloader.poison_data,
                test_dataloader=dataloader.test_data, former_model=predicted_model
            )

            server.aggregation(weight_accumulator=weight_accumulator, aggregated_model_id=aggregated_model_id,
                               clip_value=clip_value, round=round)

        former_model = copy.deepcopy(current_model)
        current_model = copy.deepcopy(server.global_model)

        logger.info(f" ")
        acc, acc_p = malicious_client.global_test(server.global_model, test_data=dataloader.test_data,
                                                  round=round,
                                                  poisoned_pattern_choose=server.params["poisoned_pattern_choose"])

        acc_list.append(acc)
        acc_p_list.append(acc_p)

        server.post_process()

        save_model(name="global_model", folder_path=server.folder_path, round=round,
                   lr=server.params["benign_lr"], save_on_round=server.params["save_on_round"],
                   model=server.global_model, ood_dataloader=dataloader.ood_data)

    plot_poisoned_acc(save_path=server.folder_path, start_round=server.params["start_round"],
                      acc=acc_list, acc_p=acc_p_list, is_save_img=True)
