import numpy as np
from scipy.io import loadmat, savemat
import sys, os
import torch

root_path= "/home/fengbo/Desktop/GTSM_3d/data/tetramesh_beam_data"

# 读取MAT文件
file_mode = ["coarse"]
data_mode = ["test","train","valid"]

for file in file_mode :
    for data in data_mode:
        if data == "train":
            n_traj = 256
        else :
            n_traj = 32
        path_current = os.path.join(root_path,file,data)

        for i_traj in range(n_traj):
            for i_step in range(32):
                mat_file_name = os.path.join(path_current ,"traj_{:06d}/{:04d}_00.mat".format(i_traj, i_step))
                npz_file_name = os.path.join(path_current ,"traj_{:06d}/{:04d}_00.npz".format(i_traj, i_step))
                mat_data = loadmat(mat_file_name)
                von_mises = mat_data['von_mises'].astype(np.float64)
                von_mises = np.array(torch.FloatTensor(von_mises))
                world_pos = mat_data['world_pos'].astype(np.float64)
                world_pos = np.array(torch.FloatTensor(world_pos))
                np.savez(npz_file_name, von_mises = von_mises,world_pos =world_pos)
                # a = np.load(npz_file_name)
                # a1 = a['von_mises']
                # a2 = a['world_pos']
                if i_step ==0:
                    mat_file_name = os.path.join(path_current ,"traj_{:06d}/infor.mat".format(i_traj))
                    npz_file_name = os.path.join(path_current ,"traj_{:06d}/infor.npz".format(i_traj))
                    mat_data = loadmat(mat_file_name)
                    body_force = mat_data["body_force"].astype(np.float64)
                    body_force = np.array(torch.FloatTensor(body_force))

                    cells = mat_data["cells"].astype(np.int64)

                    mesh_pos = mat_data["mesh_pos"].astype(np.float64)
                    mesh_pos = np.array(torch.FloatTensor(mesh_pos))

                    node_type = mat_data["node_type"].astype(np.int64).squeeze()

                    Elas_modulus = mat_data['E'].astype(np.float64)
                    Elas_modulus = np.array(Elas_modulus)
                    Poisson_rate = mat_data['nu'].astype(np.float64)
                    Poisson_rate = np.array(Poisson_rate)
                    
                    edge_index = np.sort(np.concatenate((cells[:,[0,1]],cells[:,[0,2]],cells[:,[0,3]],cells[:,[1,2]],cells[:,[1,3]],cells[:,[2,3]]),axis=0),axis = -1)
                    edge_indez_hash = edge_index[:,0:1]*1e6 + edge_index[:,1:2]
                    unique_hashes, unique_indices = np.unique(edge_indez_hash, return_index=True)
                    edge_index = edge_index[unique_indices,:]
                    edge_index = np.concatenate((edge_index,edge_index[:,[1,0]]),axis = 0)
                    np.savez(npz_file_name, body_force = body_force,cells =cells.tolist(),mesh_pos=mesh_pos,node_type=node_type,edge_index = edge_index,Elas_modulus = Elas_modulus,Poisson_rate = Poisson_rate)
