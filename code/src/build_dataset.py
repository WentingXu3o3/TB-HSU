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


# from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import numpy as np
# from clip import clip
# import requests
from sklearn.decomposition import PCA


def label_onehot(obj_labels,label_tokenize_dict):
    label_onehot = torch.tensor([label_tokenize_dict.index(obj) for obj in obj_labels])
    return label_onehot

class H3DSG_dataset(data.Dataset):
    def __init__(self,scans, object_rooms_all_scan,object_points_all_scan, global_config):
        global config
        config = global_config
        dataset = []
        all_roomtypes = set()
        self.all_roomtypes_count_dict = {}
        all_grouptypes = set()
        self.all_grouptypes_count_dict = {}
        self.object_area_dict = {}
        self.max_objects_num = 0
        label_set = set()
        self.nodes = 0
        # self.edges = 0
        for scan in scans:
            roomtype = object_rooms_all_scan[scan]['roomtype'][1]

            if roomtype not in self.all_roomtypes_count_dict:
                self.all_roomtypes_count_dict[roomtype] = 1
            else:
                self.all_roomtypes_count_dict[roomtype] += 1
                
            
            obj_area_dict = {}
            edges = 0
            for group in object_rooms_all_scan[scan]['groups'].values():
                areatype = group['areaname'][1]
                all_grouptypes.add(areatype)

                if areatype not in self.all_grouptypes_count_dict:
                    self.all_grouptypes_count_dict[areatype] = 1
                else:
                    self.all_grouptypes_count_dict[areatype] += 1

                for id in group['contents']:
                    obj_area_dict[id[0]] = areatype

                edges += len(group['contents'])
            self.nodes += len(object_rooms_all_scan[scan]['groups']) + edges + 1

            obj_points_set = []
            obj_labels = []
            obj_areas = []
            obj_labels_id = []
            obj_obb_centroid = []
            obj_obb_Axes = []
            obj_dominantNormal = []
            obj_volume = []
            obj_extend =[]
            keys = list ( object_points_all_scan[scan].keys())


            for index,(id, obj) in enumerate(object_points_all_scan[scan].items()):
 
                if obj['label'] not in ['wall','floor','ceiling','wall /other room']:
                    obj_labels.append(obj['label'])
                    label_set.add(obj['label'])
                    
                    obj_labels_id.append(id)
                    obj_points_set.append(obj['points']) 
                    # PCA_data = PCA_Method(obj['points'])[0].flatten()

               
                    # obj_pca_set.append(PCA_data)
    
                    obb_centroid = torch.tensor(obj["obb"]["centroid"])#3
                    axes_lengths = torch.tensor(obj["obb"]['axesLengths'])#3
                    normalized_axes = torch.tensor(obj["obb"]['normalizedAxes']).reshape(3, 3)
        
                    normalized_axes_scaled =  axes_lengths.unsqueeze(0) * normalized_axes  #3,3
                    normalized_axes_sumonxyz = np.abs(np.dot(np.array(obj["obb"]["axesLengths"]), np.array(obj["obb"]['normalizedAxes']).reshape(3,3).T)) #3
                    obj_extend.append(normalized_axes_sumonxyz)

                    obj_obb = torch.cat((obb_centroid - 0.5*(normalized_axes_scaled), obb_centroid + 0.5*(normalized_axes_scaled)),dim=0) #6,3
                

                    #8endpoints for each bounding box
                    corners_local = torch.tensor([[+1, +1, +1],
                                            [+1, +1, -1],
                                            [+1, -1, +1],
                                            [+1, -1, -1],
                                            [-1, +1, +1],
                                            [-1, +1, -1],
                                            [-1, -1, +1],
                                            [-1, -1, -1]]) * axes_lengths/2 #8*3
    
           
                    corners_global = torch.matmul(corners_local,normalized_axes) + obb_centroid #8*3
    
                    obj_obb_Axes.append(corners_global.tolist())

                                        # obj_obb_centroid.append(obj["obb"]["centroid"])
                    obj_obb_centroid.append(np.mean(obj['points'],axis=0))

                    self.obb_channel_len = 78
 
                    obj_obb_centroid_axes = torch.cat(((torch.tensor(obj["obb"]["centroid"])).unsqueeze(0), normalized_axes_scaled),dim=0)
                    # print(obj_obb_centroid_axes.shape) #4,3

                    obj_dominantNormal.append(obj_obb_centroid_axes.tolist())

                    # print(obj["obb"]['axesLengths'][0])
                    volume = obj["obb"]['axesLengths'][0] * obj["obb"]['axesLengths'][1] * obj["obb"]['axesLengths'][2]
                    # print(volume)
                    obj_volume.append(volume)
                    

                    if obj['label'] not in self.object_area_dict:
                        self.object_area_dict[obj['label']] = {}
                    
                    area_dict = self.object_area_dict[obj['label']]
                    
                    if obj_area_dict[id] not in area_dict:
                        area_dict[obj_area_dict[id]] = 1
                    else:
                        area_dict[obj_area_dict[id]] +=1

                    self.object_area_dict[obj['label']] = area_dict

                    try:
                        obj_areas.append(obj_area_dict[id])
                    except:
                        print("error raw scans data load")
                        print(scan)
                        print(id,obj['label'])
                        print(obj_area_dict)
                        print(object_rooms_all_scan[scan]['groups'].values())
                       
            obj_obb_Axes = torch.tensor(obj_obb_Axes)
            x_max = torch.max(torch.max(obj_obb_Axes[:,:,0], dim = -1).values).item()
            y_max = torch.max(torch.max(obj_obb_Axes[:,:,1], dim = -1).values).item()
            z_max = torch.max(torch.max(obj_obb_Axes[:,:,2], dim = -1).values).item()
            x_min = torch.min(torch.min(obj_obb_Axes[:,:,0], dim = -1).values).item()
            y_min = torch.min(torch.min(obj_obb_Axes[:,:,1], dim = -1).values).item()
            z_min = torch.min(torch.min(obj_obb_Axes[:,:,2], dim = -1).values).item()
            room_obb_8_points = torch.tensor([[x_max,y_max,z_max],
                                    [x_max,y_max,z_min],
                                    [x_max,y_min,z_max],
                                    [x_max,y_min,z_min],
                                    [x_min,y_max,z_max],
                                    [x_min,y_max,z_min],
                                    [x_min,y_min,z_max],
                                    [x_min,y_min,z_min]]
                                    ) #8,3
            room_obb_axis_lengths = torch.abs(room_obb_8_points.max(dim=0)[0] - room_obb_8_points.min(dim=0)[0])
            # Calculate the volume of the box
            room_obb_volume = room_obb_axis_lengths.prod()
            
            if len(obj_labels) > self.max_objects_num:
                self.max_objects_num = len(obj_labels)
            obj_obb_centroid = torch.tensor(obj_obb_centroid)

            all_roomtypes.add(roomtype)
            dataset.append({
                "roomtype": roomtype,
                "obj_points_set" : obj_points_set,
                # "obj_pca_set": obj_pca_set,
                "obj_labels": obj_labels,
                "obj_areas": obj_areas,
                "obj_labels_id" : obj_labels_id,
                "scanid": scan,
                "obj_obb_centroid": obj_obb_centroid,
                # "room_obb_centroid": torch.mean(obj_obb_centroid,dim=0),
                "obj_dominantNormal": torch.tensor(obj_dominantNormal),
                "obj_obb_Axes": obj_obb_Axes,
                "obj_obb_volume_raw":torch.tensor(obj_volume),
                # "obj_obb_volume": torch.tensor(obj_volume)/room_obb_volume,
                "obj_obb_volume": torch.tensor(obj_volume),
                "room_obb_8_points":room_obb_8_points,
                "room_obb_volume":room_obb_volume,
                "normalized_axes_sumonxyz":obj_extend
            })
        self.dataset = dataset
        self.config = config
        self.num_objects = config.num_objects
        self.label_set = sorted(list(label_set))
        self.all_roomtypes = sorted(list(all_roomtypes))
        self.all_grouptypes = sorted(list(all_grouptypes))
        self.HSU_info = {}
        

    def __getitem__(self, index):

        roomtype = self.dataset[index]["roomtype"]

        obj_labels_list = self.dataset[index]["obj_labels"][:] # [:] need this to do the independent copy
        obj_labels = self.dataset[index]["obj_labels"][:]
        
        
        for i in range(self.num_objects - len(obj_labels_list)):
            obj_labels.append('')
      

        obj_label_tokenize = label_onehot(obj_labels, self.HSU_info["label_tokenize_dict"])
        
        obj_areas = self.dataset[index]["obj_areas"][:]



        obj_areas_matrix = torch.zeros(self.num_objects+1) #obj_num
        # obj_areas_matrix = torch.full((self.num_objects+1,),33)
        obj_areas_matrix[0] = self.HSU_info["all_roomtypes_list"].index(roomtype)
        for idx, area in enumerate(obj_areas):
            obj_areas_matrix[idx+1] = self.HSU_info["all_grouptypes_list"].index(area)

        obj_areas_matrix = obj_areas_matrix.to(torch.int64)

        '''obb centroid and volume'''

        obj_obb = torch.zeros(self.num_objects, 5) #5
        #0,1,2:obj_obb_centroid  3:obj_obb_volume              -1:obj_obb_distance 
        # obj_obb = torch.full((self.num_objects, 4),99)
        room_centroid  = torch.mean(self.dataset[index]["obj_obb_centroid"], dim=0)



        for idx, one_scan in enumerate(self.dataset[index]["obj_obb_centroid"]):
            if config.pre_objs_meancentroid:
                obj_obb[idx][:3]= one_scan - room_centroid
                obj_obb[idx][-1] = torch.norm(one_scan - room_centroid)

            else:
                obj_obb[idx][:3]= one_scan
            obj_obb[idx][3] = self.dataset[index]["obj_obb_volume"][idx]

        room_obb = obj_obb.to(torch.float32) # 77,4
       

        return{
            "index":index,
            "scanid":self.dataset[index]["scanid"],
            "roomtype":0,
            # "obj_points":obj_points,
            "room_obb":room_obb,
            "obj_obb": 0,
            "obj_label_tokenize":obj_label_tokenize,
            "obj_labels": obj_labels,
            # "obj_labels_list":obj_labels_list,
            # "room_points":room_points,
            # "obj_labels_list_token":obj_labels_list_token,
            # "room_points_sparse_pick":room_points_sparse_pick,
            "obj_areas_matrix":obj_areas_matrix,
            # "obj_areas_token": obj_areas,
            "obj_num":len(self.dataset[index]["obj_labels"][:]),
            "obj_area_tokenized":0
        }
    def __len__(self):
        return len(self.dataset)
