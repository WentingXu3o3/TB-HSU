# MIT License

# Copyright (c) 2025 Wenting Xu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#####
#####This is the main code for the whole H3DSG project
#####
#####Created by Wenting Xu on 19/Feb/2024 Happy Chinese New Year@Wenting !

import pandas as pd
import os,json
import numpy as np
import time
from clip import clip

from torch.utils.data import DataLoader
from torch.optim import AdamW,SGD
import torch

from torch.utils.data import DataLoader
import torch.utils.data as data

import random
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from model import H3DSG_model

from DataPreparation import DP_Objects_PointCloud
import src.main_utils as main_utils
from src.main_utils import wandb_init,zero_mean
from src.build_dataset import H3DSG_dataset
from config import config
import sys

def H3DSG_Object_Groups_Prepartion(GroupingFile):
    if os.path.exists(os.path.join(os.path.dirname(__file__),'3DHSG',GroupingFile)):
        with open (os.path.join(os.path.dirname(__file__),'3DHSG',GroupingFile),'r')as f:
            scans_dict = json.load(f)
            scans = scans_dict.keys()
    else:
        with open (os.path.join(os.path.dirname(__file__),'3DHSG','Aff_Layer1_Match_dict.json'),'r')as f:
            aff_layer1 = json.load(f)
            match_aff_room= {}
            for key, value in aff_layer1.items():
                match_aff_room[value] = key
            list_aff_room = sorted(list(aff_layer1.keys()))
        with open (os.path.join(os.path.dirname(__file__),'3DHSG','Aff_Layer2_Match_dict.json'),'r')as f:
            aff_layer2 = json.load(f)
            match_aff_region= {}
            for key, value in aff_layer2.items():
                match_aff_region[value] = key
            list_aff_region = sorted(list(aff_layer2.keys()))
        with open (os.path.join(os.path.dirname(__file__),'3DHSG','3DHSG.json'),'r')as f:
            H3DSG = json.load(f)

        scans_dict = {}
        for scan in scans:
            groups = {}
            for key,values in H3DSG[scan]["regions"].items():
                groups[key] = {
                    "areaname": [list_aff_region.index(match_aff_region[values["region_specific_aff"]]),match_aff_region[values["region_specific_aff"]]],
                    "contents": values["contents"]
                }
            scans_dict[scan]={
                "roomtype": [list_aff_room.index(match_aff_room[H3DSG[scan]["roomtype"]]),match_aff_room[H3DSG[scan]["roomtype"]]],
                "groups": groups
            }
        
        with open (os.path.join(os.path.dirname(__file__),'3DHSG',f'{GroupingFile}'),'w')as f:
            json.dump(scans_dict,f,indent=4)
    # assert len(scans) == len(scans_dict)

    return scans, scans_dict

def H3DSG_Get_Objects_points_all_scan(scans):
    starttime = time.time()
    object_points_all_scan = {}
    for scan in scans:
        object_points = DP_Objects_PointCloud.data_preparation(scan)
        object_points_all_scan[scan] = object_points

    np.savez(os.path.join(os.path.dirname(__file__),'H3DSG_load_points.npz'), **object_points_all_scan)
    print("save H3DSG_load_points.npz file")
    endtime = time.time()
    print(endtime-starttime,"seconds")
    return object_points_all_scan

def Dataset_Preparation_raw(datasetname,scanname):
    if datasetname == 'H3DSG':
        scans, object_rooms_all_scan = H3DSG_Object_Groups_Prepartion('3DHSG_load_all_matched_Groups.json')

        if config.RegetData:
            object_points_all_scan = H3DSG_Get_Objects_points_all_scan(scans)
            print("Data Rebuild")
        else:
            data = np.load(os.path.join(os.path.dirname(__file__),'H3DSG_load_points.npz'),allow_pickle=True)
            # object_points_all_scan = {key: data[key].item() for key in data}
            object_points_all_scan = {}
            for idx, key in enumerate(data):
                if key != scanname:
                    nested_dict = {}
                    for nested_key in data[key].item():
                        nested_dict[nested_key] = data[key].item()[nested_key]
                    object_points_all_scan[key] = nested_dict
                else:
                    break
        assert len(object_points_all_scan) == len(scans)
        return list(object_points_all_scan.keys()), object_rooms_all_scan, object_points_all_scan


def calculate_metrics(TP, FN, FP):
    # TP is the number of true positives
    # FN is the number of false negatives
    # FP is the number of false positives
    # in room classification for each room type, take kitchen type as an example
    # TP is number of "kitchen" to be seen as kitchen
    # FN is number of "kitchen" to be seen as not kitchen
    # FP is number of "not kitchen" to be seen as kitchen
    
    recall = TP / (TP + FN + 1e-6)  # Adding a small value to avoid division by zero
    IoU = TP / (TP + FP + FN + 1e-6)  # Adding a small value to avoid division by zero

    return recall, IoU

def evaluation_2():
    model.eval()
    rm_correct_initial = 0
    at_correct_initial = 0

    all_roomtypes_list = HSU_info["all_roomtypes_list"]
    all_areatypes_list = HSU_info["all_grouptypes_list"]

    TP_rm = torch.zeros(len(all_roomtypes_list),dtype=torch.int32)
    FN_rm = torch.zeros(len(all_roomtypes_list),dtype=torch.int32)
    FP_rm = torch.zeros(len(all_roomtypes_list),dtype=torch.int32)
    
    TP_at = torch.zeros(len(all_areatypes_list),dtype=torch.int32)
    FN_at = torch.zeros(len(all_areatypes_list),dtype=torch.int32)
    FP_at = torch.zeros(len(all_areatypes_list),dtype=torch.int32)
    All_area_dict ={}
    with open (os.path.join(os.path.dirname(__file__),'H3DSG_gt.json'),'r')as f:
        scanid_groups = json.load(f)

    for idx, batch in enumerate(validation_dataloader): 

        obj_obb = batch["obj_obb"]
        obj_label_tokenize = batch["obj_label_tokenize"]
        room_obb = batch["room_obb"]

        obj_areas_matrix = batch["obj_areas_matrix"]

        len_obj_labels = batch["obj_num"]

        ground_truth_matrix = obj_areas_matrix

       
        with torch.no_grad():
           
            obj_obb = obj_obb.to(device)
            obj_label_tokenize = obj_label_tokenize.to(device)
            room_obb = room_obb.to(device)


            predicted_roomtype, predicted_areatypes,attn_matrix = model(
                len_obj_labels,
                room_obb = room_obb,
                obj_obbs = obj_obb, 
                obj_token = obj_label_tokenize, 
                cat=config.cat)

            roomtype_gt = ground_truth_matrix[:,0] #torch.Size([1, 1])
            areatype_gt = ground_truth_matrix[:,1:] # print(areatype_gt.shape) #torch.Size([1, 77])
           
            rm_probs = predicted_roomtype.softmax(dim=-1).cpu()# batchsize, 78,21
            at_probs = predicted_areatypes.softmax(dim=-1).cpu() ## batchsize, 78,27

        rm_index = torch.max(rm_probs,dim=-1)
        rm_correct_initial += torch.sum(rm_index.indices == roomtype_gt).int()

        at_index = torch.max(at_probs, dim=-1) #batchsize, 77

        correct_predictions = torch.sum(at_index.indices[:,:len_obj_labels] == areatype_gt[:,:len_obj_labels]).float().item()
        total_predictions = (areatype_gt.size(0)*len_obj_labels)
        
        at_correct_initial += correct_predictions / total_predictions
           
        if config.print_log:
            print("\n\n")
            print(scans.index(validation_dataset.dataset[idx]["scanid"]),validation_dataset.dataset[idx]["scanid"],"RoomtypePredict:",all_roomtypes_list[rm_index.indices[0]],"GT:",all_roomtypes_list[roomtype_gt[0]])
        
        Area_dict = {}
        for i in range(len_obj_labels):
            at_index_group = HSU_info["all_grouptypes_list"][at_index.indices[0, i]]
            gt_group = HSU_info["all_grouptypes_list"][areatype_gt[0, i]]
            area_info = (validation_dataset.dataset[idx]["obj_labels_id"][i], validation_dataset.dataset[idx]["obj_labels"][i],at_index.values[0,i].item())
            
            # Check if the group key exists in the dictionary, if not, initialize it with an empty list
            if at_index_group not in Area_dict:
                Area_dict[at_index_group] = []
            
            # Append the area info to the corresponding group key
            Area_dict[at_index_group].append(area_info)
            if config.print_log:
                print(validation_dataset.dataset[idx]["obj_labels_id"][i], validation_dataset.dataset[idx]["obj_labels"][i], "Predict:", at_index_group, "GT:", gt_group)
        if config.print_log:
            print("AT Accuracy The Scan:",'\033[31m'+str((correct_predictions / total_predictions).item())+'\033[0m')
        
        '''rm mIoU'''
        for i in range(len(all_roomtypes_list)):
            # Calculate TP, FP, FN for each room type
            TP_rm[i] += ((roomtype_gt == i) & (rm_index.indices == i)).sum().item()
            FN_rm[i] += ((roomtype_gt == i) & (rm_index.indices != i)).sum().item()
            FP_rm[i] += ((roomtype_gt != i) & (rm_index.indices == i)).sum().item()
        
        
        '''at mIoU'''
        # print(at_index.indices.shape) #1,77
        # print(len(all_areatypes_list)) #27
        # print(len_obj_labels) #21
        for i in range(len(all_areatypes_list)):
            at_gt = areatype_gt[:,:len_obj_labels]
            at_pred = at_index.indices[:,:len_obj_labels]

            TP_at[i] += ((at_gt == i) & (at_pred == i)).sum().item()
            FN_at[i] += ((at_gt == i) & (at_pred != i)).sum().item()
            FP_at[i] += ((at_gt != i) & (at_pred == i)).sum().item()


        
        # print(all_roomtypes_list[rm_index.indices[0]],":")
        # # print(Area_dict)
        if config.print_log:
            print("\n")
            for key, value in Area_dict.items():
                print(f"{key}\n: {value}")
        # print(batch["scanid"])

        All_area_dict[batch["scanid"][0]] = {
            "Layer1":all_roomtypes_list[rm_index.indices[0]],
            "Layer2":Area_dict,
            "GT_Layer1":scanid_groups[batch["scanid"][0]]["Layer1"],
            "GT_Layer2":scanid_groups[batch["scanid"][0]]["Layer2"]
            }
        
    rm_recall, rm_IoU = calculate_metrics(TP_rm, FN_rm, FP_rm)
    at_recall, at_IoU = calculate_metrics(TP_at, FN_at, FP_at)  
    
    eval_roomtypes_lits = validation_dataset.all_roomtypes # for test set there are only 13 types
    TYPE_VALID_RM_CLASS_IDS = [all_roomtypes_list.index(cls) for cls in eval_roomtypes_lits]

    rm_recall = rm_recall[TYPE_VALID_RM_CLASS_IDS]
    rm_IoU = rm_IoU[TYPE_VALID_RM_CLASS_IDS]

    eval_grouptypes_lits = validation_dataset.all_grouptypes # 
    TYPE_VALID_AT_CLASS_IDS = [all_areatypes_list.index(cls) for cls in eval_grouptypes_lits]

    at_recall = at_recall[TYPE_VALID_AT_CLASS_IDS]
    at_IoU = at_IoU[TYPE_VALID_AT_CLASS_IDS]


    if config.print_log:
        print("\n")
        print("Validation Results:")
        print("     Room_Accuracys:",rm_correct_initial.item()/len(validation_dataloader))
        print("     AT_Accuracys:",at_correct_initial/len(validation_dataloader),"\n")
    
    mean_accuracy = (rm_correct_initial.item()/len(validation_dataloader)+at_correct_initial/len(validation_dataloader))/2.0
    
    return at_correct_initial/len(validation_dataloader),\
        rm_correct_initial.item()/len(validation_dataloader),\
        mean_accuracy,\
        rm_recall,\
        rm_IoU,\
        at_recall,\
        at_IoU,\
        All_area_dict

def get_train_test_set():
    with open (os.path.join(os.path.dirname(__file__),'3DHSG/train.txt'),'r')as f:
        train_dataset =  f.read().splitlines()
    with open(os.path.join(os.path.dirname(__file__),'3DHSG/test.txt'),'r')as f:
        validation_dataset = f.read().splitlines()

    return train_dataset, validation_dataset
def random_seed_settle(config):
    random.seed(config.random_seed) 
    np.random.seed(config.random_seed) # numpy random seed
    torch.manual_seed(config.random_seed) 
    torch.autograd.set_detect_anomaly(True)
    os.environ["PYTHONHASHSEED"] = str(config.random_seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    config_file_path = os.path.join(os.path.dirname(__file__), 'config','config.json')
    config = config.GetConfig(config_file_path)

    Project_name, name = wandb_init(config)
    wandb.init(project=Project_name, config = config.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    random_seed_settle(config)


    if config.dataset=='H3DSG':
        scans, object_rooms_all_scan, object_points_all_scan = Dataset_Preparation_raw(config.dataset,config.scanname)
        train_dataset, validation_dataset = get_train_test_set()

        train_dataset = H3DSG_dataset(train_dataset, object_rooms_all_scan, object_points_all_scan,config)
        validation_dataset = H3DSG_dataset(validation_dataset, object_rooms_all_scan, object_points_all_scan,config)
        
        all_grouptypes = sorted(set(validation_dataset.all_grouptypes) | set(train_dataset.all_grouptypes))
        all_roomtypes = sorted(set(validation_dataset.all_roomtypes) | set(train_dataset.all_roomtypes))

        combined_labels = set(train_dataset.label_set) | set(validation_dataset.label_set)
        label_tokenize_dict = list(combined_labels)
        label_tokenize_dict.sort()
        label_tokenize_dict = label_tokenize_dict + [''] #keep a space for padding token

        HSU_info = {
            "all_grouptypes_list":all_grouptypes,
            "all_roomtypes_list":all_roomtypes,
            "label_tokenize_dict":label_tokenize_dict
        }

        train_dataset.HSU_info = HSU_info
        validation_dataset.HSU_info = HSU_info

        # nnnmodel.baseline(train_dataset,validation_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle= True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle= False)

        # gptEval(validation_dataset)
        
        model = H3DSG_model(            
            config,
            len(all_roomtypes),
            len(all_grouptypes),
            embed_dim = config.embed_dim,
            num_objects =  config.num_objects,
            text_embedding_context_length = config.text_embedding_context_length,
            text_embedding_transformer_width = config.embed_dim,
            text_embedding_layers = config.text_embedding_layers,
            transformer_width = config.transformer_width,
            transformer_layers = config.transformer_layers,
            semantic_embedding_is = config.semantic_embedding_is,
            positional_embedding_is = config.positional_embedding_is, 
            text_embedding_vocab_size = len(label_tokenize_dict),
            zero_shot = config.zero_shot,
            dropout = config.dropout,
            baseline_is = config.baseline_is,
            baseline_model = config.baseline_model
            ).to(device)
        # print("\n",model)


        optimizer = SGD(model.parameters(), config.lr, momentum=config.mm, weight_decay=config.wd)
    

        maxaccuracy = 0

        loss_function = torch.nn.CrossEntropyLoss()

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n\nNum_Parameters for the H3DSG model ", '\033[31m'+str(num_parameters)+'\033[0m',"\n\n")

        print("Ready to Train?")
        
        for epoch in range(config.num_epochs):
            sys.stdout.write(f'\rEpoch {epoch+1}/{config.num_epochs}')
            sys.stdout.flush()
            if epoch == 0:
                if config.print_log:
                    print("\n****************** Evaliation Before Train ***************")
                # evaluation_2(epoch=epoch)

            model.train()
            total_loss = 0.0
            if config.print_log:
                print(f"****************** Epoch:{epoch+1} Train *******************")
            training_accuracy = 0
            rm_correct_initial = 0
            at_correct_initial = 0
            for batch in train_dataloader:

                obj_obb = batch["obj_obb"].to(device)
                obj_label_tokenize = batch["obj_label_tokenize"].to(device)
                len_obj_labels = batch["obj_num"]
                scanids = batch["scanid"]
  
                obj_areas_matrix = batch["obj_areas_matrix"].to(device)

                roomtype_gt = obj_areas_matrix[:,0]
                areatype_gt = obj_areas_matrix[:,1:]

                room_obb = batch["room_obb"].to(device)
            

                logits_rm, logits_at, attn_matrix  = model(
                    len_obj_labels,
                    # roomtype = roomtype_tokenize, 
                    room_obb = room_obb,
                    obj_obbs = obj_obb, 
                    obj_token = obj_label_tokenize,
                    # obj_areas_token = obj_areas_token, 
                    cat=config.cat,
                    train_mode = True)
                loss_rm = loss_function(logits_rm,roomtype_gt)

                logits_at = logits_at.reshape(logits_at.size(0)*logits_at.size(1), logits_at.size(2))

                areatype_gt = areatype_gt.reshape(areatype_gt.size(0)*areatype_gt.size(1))

                loss_at = loss_function(logits_at,areatype_gt)
                

                loss = loss_rm/(loss_rm+loss_at)*loss_rm + loss_at/(loss_rm+loss_at)*loss_at

                optimizer.zero_grad()  
        
                loss.backward()

                optimizer.step()

                total_loss += loss.item()


                
                """train accuracy"""
                rm_probs = logits_rm.softmax(dim=-1)  ## batchsize,21
                at_probs = logits_at.softmax(dim=-1) ## batchsize, 78,27 # print(probs.shape) # torch.Size([1, 78, 27])
                
                rm_index = torch.max(rm_probs,dim=-1)
                correct_predictions_rm = torch.sum(rm_index.indices == roomtype_gt).int()


                rm_correct_initial +=  correct_predictions_rm / roomtype_gt.numel()
                

                at_index = torch.max(at_probs, dim=-1) #batchsize*77
               
                at_index_indices_reshape = at_index.indices.view(obj_areas_matrix.size(0),at_index.indices.size(0)//obj_areas_matrix.size(0))
                at_index_values_reshape = at_index.values.view(obj_areas_matrix.size(0),at_index.indices.size(0)//obj_areas_matrix.size(0))
                areatype_gt=areatype_gt.view(obj_areas_matrix.size(0), areatype_gt.size(0)//obj_areas_matrix.size(0))
 
                correct_predictions_at = 0
                for idx, len_obj_label in enumerate(len_obj_labels):
                    correct_predictions_at += torch.sum(at_index_indices_reshape[idx,:len_obj_labels[idx]] == areatype_gt[idx,:len_obj_labels[idx]]).float()

                total_predictions = torch.sum(len_obj_labels)
                at_correct_initial += correct_predictions_at / total_predictions

               
            '''accuracy for test set'''
            at_accuracy, room_accuracy, mean_accuracy, rm_recall, rm_IoU, at_recall, at_IoU , Area_dict = evaluation_2(epoch = epoch+1)

            average_loss = total_loss / len(train_dataloader)
            train_at_accuracy = at_correct_initial/len(train_dataloader)
            if config.print_log:
                print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {average_loss}, Rm: {rm_correct_initial/len(train_dataloader)}, At:{train_at_accuracy}")
            
            #visualization with wandb
            wandb.log({'loss':average_loss,'AT_accuracy':at_accuracy,'AT_mIoU':at_IoU.mean().item(),'RT_accuracy':room_accuracy,'RT_mIoU':rm_IoU.mean().item()})
            
