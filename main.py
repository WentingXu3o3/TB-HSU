# MIT License

# Copyright (c) [year] [fullname]

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


import os,json
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from model import H3DSG_model
from src.build_dataset import H3DSG_dataset
from config import config


def H3DSG_Object_Groups_Prepartion(GroupingFile):
    with open(os.path.join(os.path.dirname(__file__),'3DHSG','test.txt'),'r')as f:
        scans = f.read().splitlines()
    if os.path.exists(os.path.join(os.path.dirname(__file__),'3DHSG',GroupingFile)):
        with open (os.path.join(os.path.dirname(__file__),'3DHSG',GroupingFile),'r')as f:
            scans_dict = json.load(f)
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
        with open (os.path.join(os.path.dirname(__file__),'3DHSG_dataset','3DHSG_test.json'),'r')as f:
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
        
        with open (os.path.join(os.path.dirname(__file__),'3DHSG',GroupingFile.json),'w')as f:
            json.dump(scans_dict,f,indent=4)
    assert len(scans) == len(scans_dict)

    return scans, scans_dict
 
def Dataset_Preparation_raw(datasetname):
     if datasetname == '3DHSG':
        scans, object_rooms_all_scan = H3DSG_Object_Groups_Prepartion('3DHSG_load_test_mathced_Groups.json')

        data = np.load(os.path.join(os.path.dirname(__file__),'3DHSG/H3DSG_load_test_points.npz'),allow_pickle=True)

        object_points_all_scan = {}
        for idx, key in enumerate(data):
            if key in scans:
                nested_dict = {}
                for nested_key in data[key].item():
                    nested_dict[nested_key] = data[key].item()[nested_key]
                object_points_all_scan[key] = nested_dict
            else:
                print(f"Scan {key} not in the list of scans")
        return scans, object_rooms_all_scan, object_points_all_scan

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
    # mIoU = IoU.mean().item()  # Mean IoU

    # print(f"Recall per class: {recall}")
    # print(f"IoU per class: {IoU}")
    # print(f"Mean IoU: {mIoU}")

    return recall, IoU

def test(model):
    state_dict_path = os.path.join(os.path.dirname(__file__),'results','TB_HSU.pt')
    model.load_state_dict(torch.load(state_dict_path))
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

    for idx, batch in enumerate(test_dataloader): 
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
            if not config.zero_shot:
                predicted_roomtype, predicted_areatypes, attn_matrix = model(
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
        
        at_index = torch.max(at_probs, dim=-1) 

        correct_predictions = torch.sum(at_index.indices[:,:len_obj_labels] == areatype_gt[:,:len_obj_labels]).float().item()
        total_predictions = (areatype_gt.size(0)*len_obj_labels)
        
        at_correct_initial += correct_predictions / total_predictions
        
        if config.print_log:
            print("\n\n")
            print(scans.index(test_dataset.dataset[idx]["scanid"]),test_dataset.dataset[idx]["scanid"],"RoomtypePredict:",all_roomtypes_list[rm_index.indices[0]],"GT:",all_roomtypes_list[roomtype_gt[0]])
        
        Area_dict = {}
        for i in range(len_obj_labels):
            at_index_group = HSU_info["all_grouptypes_list"][at_index.indices[0, i]]
            gt_group = HSU_info["all_grouptypes_list"][areatype_gt[0, i]]
            area_info = (test_dataset.dataset[idx]["obj_labels_id"][i], test_dataset.dataset[idx]["obj_labels"][i],at_index.values[0,i].item())
            
            # Check if the group key exists in the dictionary, if not, initialize it with an empty list
            if at_index_group not in Area_dict:
                Area_dict[at_index_group] = []
            
            # Append the area info to the corresponding group key
            Area_dict[at_index_group].append(area_info)
            if config.print_log:
                print(test_dataset.dataset[idx]["obj_labels_id"][i], test_dataset.dataset[idx]["obj_labels"][i], "Predict:", at_index_group, "GT:", gt_group)
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
            "Layer2":Area_dict
            }
       

    rm_recall, rm_IoU = calculate_metrics(TP_rm, FN_rm, FP_rm)
    at_recall, at_IoU = calculate_metrics(TP_at, FN_at, FP_at)  
    
    eval_roomtypes_lits = test_dataset.all_roomtypes 
    TYPE_VALID_RM_CLASS_IDS = [all_roomtypes_list.index(cls) for cls in eval_roomtypes_lits]

    rm_recall = rm_recall[TYPE_VALID_RM_CLASS_IDS]
    rm_IoU = rm_IoU[TYPE_VALID_RM_CLASS_IDS]

    eval_grouptypes_lits = test_dataset.all_grouptypes 
    TYPE_VALID_AT_CLASS_IDS = [all_areatypes_list.index(cls) for cls in eval_grouptypes_lits]

    at_recall = at_recall[TYPE_VALID_AT_CLASS_IDS]
    at_IoU = at_IoU[TYPE_VALID_AT_CLASS_IDS]


    if config.print_log:
        print("\n")
        print("Validation Results:")
        print("     Room_Accuracys:",rm_correct_initial.item()/len(test_dataloader))
        print("     AT_Accuracys:",at_correct_initial/len(test_dataloader),"\n")
    # print("Loss:",at_loss.item()/len(validation_dataloader),"\n\n")
    mean_accuracy = (rm_correct_initial.item()/len(test_dataloader)+at_correct_initial/len(test_dataloader))/2.0

    return at_correct_initial/len(test_dataloader),\
        rm_correct_initial.item()/len(test_dataloader),\
        mean_accuracy,\
        rm_recall,\
        rm_IoU,\
        at_recall,\
        at_IoU,\
        All_area_dict

if __name__ == '__main__':
    config_file_path = os.path.join(os.path.dirname(__file__), 'config','config.json')
    config = config.GetConfig(config_file_path)
    # the config file is set to run TB-HSU with positional embedding to reproduce the multi-results.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    if config.dataset=='3DHSG':
        scans, object_rooms_all_scan, object_points_all_scan = Dataset_Preparation_raw(config.dataset)
       
        test_dataset = H3DSG_dataset(scans, object_rooms_all_scan, object_points_all_scan,config)
        
        all_grouptypes = ['appliance_area', 'bathing_area', 'bedding_area', 'changing_area', 'coffee_bar', 'commode_area', 'decoration_area', 'dining_area', 'entrance_area', 'fireplace_area', 'kitchen_area', 'lightening_area', 'makingup_area', 'nightstand_area', 'office_area', 'others', 'rack_area', 'shelf_area', 'sink_area', 'sitting_area', 'storage_area', 'supporting_area', 'table_area', 'toilet_area', 'tv_area', 'wardrobe_area', 'window_area']
        all_roomtypes = ['bathroom', 'bedroom', 'kitchen', 'livingroom', 'lobby', 'meetingroom', 'nursery', 'office', 'others', 'restaurant', 'storeroom', 'studio']
        label_tokenize_dict = ['air conditioner', 'armchair', 'baby bed', 'baby changing unit', 'baby seat', 'backpack', 'bag', 'balcony', 'bar', 'basket', 'bath cabinet', 'bathtub', 'bean bag', 'bed', 'bench', 'bidet', 'bin', 'blanket', 'blinds', 'board', 'boiler', 'book', 'books', 'bookshelf', 'boots', 'bottle', 'box', 'boxes', 'bucket', 'cabinet', 'candle', 'carpet', 'cart', 'chair', 'cleaning brush', 'cleanser', 'clock', 'closet', 'clothes', 'clothes dryer', 'clutter', 'coffee machine', 'coffee table', 'column', 'commode', 'container', 'couch', 'couch table', 'counter', 'cube', 'cup', 'cupboard', 'curtain', 'cushion', 'cutting board', 'decoration', 'desk', 'device', 'dishes', 'dispenser', 'door', 'door /other room', 'doorframe', 'drawer', 'dressing table', 'exhaust hood', 'extractor fan', 'fan', 'fireplace', 'flower', 'folder', 'frame', 'furniture', 'hair dryer', 'handbag', 'hanger', 'hanging cabinet', 'heater', 'humidifier', 'item', 'items', 'juicer', 'kettle', 'kids bicycle', 'kitchen appliance', 'kitchen cabinet', 'kitchen counter', 'kitchen hood', 'kitchen item', 'kitchen towel', 'ladder', 'lamp', 'laptop', 'laundry basket', 'letter', 'light', 'loft bed', 'luggage', 'magazine files', 'magazine rack', 'menu', 'microwave', 'mirror', 'monitor', 'napkins', 'nightstand', 'object', 'organizer', 'ottoman', 'oven', 'pack', 'painting', 'pan', 'paper', 'paper holder', 'paper towel dispenser', 'papers', 'pc', 'photo frame', 'photos', 'picture', 'pile of books', 'pile of candles', 'pillar', 'pillow', 'plant', 'planter', 'plate', 'player', 'podest', 'poster', 'pot', 'price tag', 'printer', 'puf', 'puppet', 'rack', 'radio', 'rail', 'refrigerator', 'rolling pin', 'round table', 'rug', 'salt', 'shelf', 'shoe commode', 'shoe rack', 'shoes', 'showcase', 'shower', 'shower curtain', 'side table', 'sign', 'sink', 'soap dispenser', 'sofa', 'stairs', 'stand', 'stool', 'storage', 'storage container', 'stove', 'stuffed animal', 'suitcase', 'table', 'table lamp', 'telephone', 'toilet', 'toilet paper', 'toilet paper dispenser', 'toilet paper holder', 'toiletry', 'towel', 'trash can', 'trashcan', 'treadmill', 'tv', 'tv stand', 'vacuum cleaner', 'vase', 'wall frame', 'wardrobe', 'wardrobe door', 'washbasin', 'washing machine', 'water heater', 'whiteboard', 'window', 'windowsill', 'wood box', 'xbox', '']
        # print(len(label_tokenize_dict)) #191 +3 
        # input()
        HSU_info = {
            "all_grouptypes_list":all_grouptypes,
            "all_roomtypes_list":all_roomtypes,
            "label_tokenize_dict":label_tokenize_dict
        }

        test_dataset.HSU_info = HSU_info

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle= False)
        
        model = H3DSG_model(            
            config,
            len(all_roomtypes),
            len(all_grouptypes),
            embed_dim = config.embed_dim,
            num_objects =  config.num_objects,
            transformer_width = config.transformer_width,
            transformer_layers = config.transformer_layers,
            semantic_embedding_is = config.semantic_embedding_is,
            positional_embedding_is = config.positional_embedding_is, 
            text_embedding_vocab_size = len(label_tokenize_dict),
            zero_shot = config.zero_shot,
            dropout = config.dropout,
            ).to(device)
        print("\n",model)
        
        at_accuracy, room_accuracy, mean_accuracy, rm_recall, rm_IoU, at_recall, at_IoU , Area_dict = test(model)
        # with open (os.path.join(os.path.dirname(__file__),'results',f'TB_HSU_HSG.json'),'w') as f:
        #     json.dump(Area_dict,f,indent=4)
        print('AT_accuracy:',at_accuracy,'AT_mIoU:',at_IoU.mean().item(),'RT_accuracy:',room_accuracy,'RT_mIoU:',rm_IoU.mean().item())