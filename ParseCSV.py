import pandas as pd
import os
import open3d as o3d
import time
import numpy as np
import h5py

# df = pd.read_csv('../../100KFilesInfo.csv')
# df = df.drop(columns=['Unnamed: 0'])

# to_drop = []
# for i, fn in enumerate(df['File']):
#     try:
#         f = open('../../STEP_files/' + fn)
#     except:
#         to_drop.append(i)
# print(len(to_drop))
# df = df.drop(to_drop)
# print(len(df))

n_bad = 0
c = 0
base = "../../STL_files/"
for root, dirs, files in os.walk(base):
    for file in sorted(files):
        f = base + file
        mesh=o3d.io.read_triangle_mesh(f)
        time.sleep(0.5)
        if not mesh.has_triangles():
            n_bad += 1
            continue

        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
        area=sum(cluster_area)
        #x=int(area//4)
        n_pts = 2048
        target = mesh.sample_points_uniformly(number_of_points=n_pts)
        # target.estimate_normals()
        
        points = np.asarray(target.points)
        # normals = np.asarray(target.normals)
        print(c)
        c += 1
        if c == 100:
            break
        # pc_path = file + ".xyz"
        # o3d.io.write_point_cloud(pc_path, target)
print(n_bad)
print(c)


