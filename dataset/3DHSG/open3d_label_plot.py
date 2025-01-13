import open3d as o3d
# o3d.visualization.gui.initialize()
#open3d.visualization.O3DVisualizer is able to plot label


import numpy as np
import os,json
from plyfile import PlyData
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

with open(os.path.join(os.path.dirname(__file__),'global_id.json'),'r') as f:
    gid_labels = json.load(f)


with open (os.path.join(os.path.dirname(__file__),'ScanIDGroups.json'),'r')as f:
    contents = json.load(f)

restart_scanid = input("ReStart With : ")



for scanid, content in contents.items():
    if int(scanid) < int(restart_scanid):
        continue
    else:

        scanname = content["scanname"]
        vertices, faces = read_mesh(f"/Volumes/sambashare/dsg_3d/data/3RScanAll/AllData/{scanname}/labels.instances.align.annotated.v2.ply")


        # vertices, faces = read_mesh("/Volumes/sambashare/dsg_3d/data/3RScanAll/AllData/0ad2d382-79e2-2212-98b3-641bf9d552c1/labels.instances.align.annotated.v2.ply")
        # vertices, faces = read_mesh("/Volumes/sambashare/dsg_3d/data/3RScanAll/AllData/4fbad31e-465b-2a5d-84b7-c0ddea978db4/labels.instances.align.annotated.v2.ply")

        # vertices, faces = read_mesh("data/3RScanAll/AllData/0ad2d382-79e2-2212-98b3-641bf9d552c1/labels.instances.align.annotated.v2.ply")
        # vertices, faces = read_mesh("data/3RScanAll/AllData/0cac75b1-8d6f-2d13-8c17-9099db8915bc/labels.instances.align.annotated.v2.ply")
        points = vertices[:, :3]  # Extracting only the XYZ coordinates\
        colors = vertices[:, 3:6] / 255.0
        # Create Open3D point cloud object
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)


        # Assign colors to segments

        # print(colors.shape)
        gids = vertices[:,7] # global_id
        # print(np.where(gids==0))


        segments = vertices[:, 6]#objectsid 
        # print(segments)
        uni_segments = np.unique(segments)
        # print("uni_segments:", uni_segments)



        segmented_point_clouds = []
        # segmented_point_clouds = o3d.visualization.gui.Window()
        ###
        ################ initialize() #################
        # app = o3d.visualization.gui.Application.instance()
        o3d.visualization.gui.Application.instance.initialize()
        # o3d.visualization.gui.Application.initialize(app)
        window = o3d.visualization.gui.Application.instance.create_window(f"Scene_{scanid}",1024,768)
        # window = app.create_window("scene",1024,768)
        view3d = o3d.visualization.gui.SceneWidget()
        # window.set_title(f"Scene_{scanid}")  # Set the name/title here
        view3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
        window.add_child(view3d)


        for object_id in uni_segments[1:]: # no idea why object_id has 0 while semseg.json dont have it.
            # print(object_id)
            
            mask = segments == object_id #mask is all points index where object_id == segments.
            segmented_points = points[mask]
            segmented_color = colors[mask]
            # Ensure segmented_color is in the correct format and dtype
            segmented_color = segmented_color.astype(np.float32)  # Convert to float32 if not already
            segmented_gid = gids[mask]
            # print(segmented_gid)
            # segmented_gid_label = gid_labels[segmented_gid[0]]
            # segmented_color = vertices[np.where(vertices[:,6]==object_id)[0],0:3]

            segment_cloud = o3d.geometry.PointCloud()
            segment_cloud.points = o3d.utility.Vector3dVector(segmented_points)
            segment_cloud.colors = o3d.utility.Vector3dVector(segmented_color)

            # view3d = o3d.visualization.gui.SceneWidget()
            # view3d.scene = o3d.visualization.rendering.Open3DScene(win.renderer)
            view3d.scene.add_geometry(f"{object_id}", segment_cloud, o3d.visualization.rendering.MaterialRecord())
        
            
            # segmented_point_clouds = o3d.visualization.O3DVisualizer()
            # segmented_point_clouds.add_geometry(segment_cloud)
        
            # segmented_point_clouds.append(segment_cloud)

            
            # Adding label annotations for each segment
            label = gid_labels[str(int(segmented_gid[0]))]
            if label not in ["wall","ceiling","floor"]:
                # Adding label annotations for each segment
                label = gid_labels[str(int(segmented_gid[0]))]
                # print(label)
                label = str(int(object_id))+str(label)
                label_position = np.mean(segmented_points, axis=0)  # Position of label (e.g., centroid of the segment)
                # label_position = label_position.astype(np.float32).flatten()

                view3d.add_3d_label(label_position,label)
                print(label,label_position)
            # segmented_point_clouds.append(label_annotation)
            # win.add_child(view3d)

        # Visualize segmented point clouds
        # o3d.visualization.draw_geometries(segmented_point_clouds)
        # segmented_point_clouds.show()
        o3d.visualization.gui.Application.instance.run()
        # o3d.visualization.gui.Application.instance.close()

    
    break

