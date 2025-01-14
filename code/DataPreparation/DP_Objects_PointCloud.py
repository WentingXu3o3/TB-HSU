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
#####This is objects_pointcloud preparation for H3DSG project
#####
#####Created by Wenting Xu on 19/Feb/2024 Happy Chinese New Year@Wenting !

import os,json
from plyfile import PlyData
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))  # Adjust the number of '..' as needed
from utils import util_ply


PC_path = 'data/3RScanAll/AllData'

def data_preparation(scan):
    # DP_Objects_PointCloud(scans)
    # PC_path = 'data/3RScanAll/AllData'
    # Scan_Objects_PC = {}
   
    with open(os.path.join(PC_path,scan,'semseg.v2.json'),'r')as f:
        contents = json.load(f)

    read_object_dict = util_ply.read_objects_points(os.path.join(PC_path, scan,'labels.instances.align.annotated.v2.ply'))
    # print(object_dict.keys())
    '''
        obj_dict[object_id]={
        "points":object_points,
        "color":object_color,
        "gid":object_gid
    }
    '''

    obj_label = {}
    obj_obb = {}
    obj_dominantNormal = {}
    for object in contents["segGroups"]:
        obj_label[object["objectId"]] = object["label"]
        obj_obb[object["objectId"]] = object["obb"]
        obj_dominantNormal[object["objectId"]] = object["dominantNormal"]
    obj_label = {key: obj_label[key] for key in sorted(obj_label.keys())}
    obj_obb = {key: obj_obb[key] for key in sorted(obj_obb.keys())}
    obj_dominantNormal = {key: obj_dominantNormal[key] for key in sorted(obj_dominantNormal.keys())}

    object_dict = {}
    for obj_id in obj_label:
        read_object_dict[obj_id]
        object_dict[obj_id] = { **read_object_dict[obj_id], 
                               "label": obj_label[obj_id],
                               "obb": obj_obb[obj_id],
                               "dominantNormal" : obj_dominantNormal[obj_id]
                               }
    # print(object_dict)
    return object_dict

    # Scan_Objects_PC[scan] = object_dict
    # print(Scan_Objects_PC)
# data_preparation()
    
if __name__ == '__main__':
    data_preparation()
