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

# This is the utils for H3DSG
# By Wenting Xu @ 20/Feb/2024

from plyfile import PlyData
import numpy as np
import trimesh

def read_mesh(point_file):
    """
    read the point 
    ply
    format ascii 1.0
    element vertex 59970
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property ushort objectId
    property ushort globalId
    property uchar NYU40
    property uchar Eigen13
    property uchar RIO27
    element face 90747
    property list uchar uint vertex_indices
    end_header
    """
    with open(point_file, 'rb') as f:
        plydata = PlyData.read(f)
        #print(plydata)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape = [num_verts,11],dtype = np.float32)
        #print(vertices)#(114805, 11)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
        vertices[:,6] = plydata['vertex'].data['objectId']
        vertices[:,7] = plydata['vertex'].data['globalId']
        vertices[:,8] = plydata['vertex'].data['NYU40']
        vertices[:,9] = plydata['vertex'].data['Eigen13']
        vertices[:,10] = plydata['vertex'].data['RIO27']
    #print(vertices.shape)
    return vertices, plydata['face']

def trimesh_read_mesh(point_file):
    plydata = trimesh.load((point_file), process=False)
    points = np.array(plydata.vertices)
    ply_raw = 'ply_raw' if 'ply_raw' in plydata.metadata else '_ply_raw'
    
    data = plydata.metadata[ply_raw]['vertex']['data']
    '''
    data = {
    'x': array
    'y'
    'z'
    'red'
    'green'
    'blue'
    'objectId'
    'globalId'
    'NYU40'
    'Eigen13'
    'RIO27'
    }
    '''
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return points, data['objectId']


def read_objects_points(point_file):
    vertices, _ = read_mesh(point_file)
    points = vertices[:, :3]  # Extracting only the XYZ coordinates\
    colors = vertices[:, 3:6] / 255.0
    gid = vertices[:,7]
    objectsId_list = np.unique(vertices[:,6]) # Unique objectId to a list
    # print(objectsId_list)
    obj_dict = {}
    for object_id in objectsId_list:
        mask = vertices[:,6] == object_id
        # print(mask)
        object_points = points[mask]
        object_color = colors[mask][0].astype(np.float32)  # Convert to float32 if not already
        # print(vertices[mask[0],7])
        object_gid = gid[mask][0] # data['globalId']
        # print(object_gid)
        obj_dict[int(object_id)]={
            "points":object_points,
            "color":object_color,
            "gid":object_gid
        }
        # print(obj_dict)
        # input()
    # print(obj_dict)
    return obj_dict

if __name__ == '__main__':
    read_objects_points('data/3RScanAll/AllData/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/labels.instances.annotated.v2.ply')
