import numpy as np
from scipy.io import savemat, loadmat
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

root_path = "/home/zheleipan/LAMP_3d/results/evo-3d_2024-06-24"
scale_factor = 10

def update(frame): 
        
    # 创建一个图形
    # 创建一个包含两个子图的绘图窗口    
    ax1.clear()
    ax2.clear()
    pos1 = ground_truth[frame]
    face1 = ground_truth_face[frame]
    # ax0.set_axis_off()
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([0, 35])
    ax2.set_zlim([-5, 5])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(30, 40)
    ax1.plot_trisurf(pos1[:, 3], pos1[:, 4],face1, pos1[:, 5], shade=True, linewidth = 0.5, edgecolor = 'grey', color="cyan")    
    ax1.set_title('Surface Plot of Ground Trurh')

    pos2 = ground_pred[frame]
    face2 = ground_pred_face[frame]
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([0, 35])
    ax2.set_zlim([-5, 5])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(30, 40)
    ax2.plot_trisurf(pos2[:, 3], pos2[:, 4],face2, pos2[:, 5], shade=True, linewidth = 0.5, edgecolor = 'grey', color="cyan")   
    ax2.set_title('Surface Plot of RLGT')
    fig.suptitle('The trajectory of the paper at step {}'.format(frame), fontsize=16)

    return fig

for i_test in range(50):
    filename = os.path.join(root_path,"test_traj_{:06d}.mat".format(i_test)) 
    data = loadmat(filename)
    Truth_history =np.array(data["Truth_his"],dtype = np.float64)
    Truth_pred = np.array(data["Truth_pred"],dtype = np.float64)
    Net_pred =  np.array(data["net_pred"],dtype = np.float64)

    Truth_history_stress =np.array(data["stress_his"],dtype = np.float64)
    Truth_pred_stress = np.array(data["Treth_streee_pred"],dtype = np.float64)
    Net_pred_stress = np.array(data["net_stress_pred"],dtype = np.float64).transpose(0,2,1)

    face_history = np.array(data["face_his"],dtype = np.int32)
    face_ground_truth = np.array(data["Truth_face_pred"],dtype = np.int32)
    face_ground_pred = np.array(data["net_face_pred"],dtype = np.int32)

    ground_truth = []
    ground_truth_stress = []
    ground_pred = []
    ground_pred_stress = []
    ground_truth_face = []
    ground_pred_face = []

    for i_array in range (Truth_history.shape[0]):
        ground_truth.append(Truth_history[i_array,:,:])
        ground_pred.append(Truth_history[i_array,:,:])

        ground_truth_stress.append(Truth_history_stress[i_array,:,:])
        ground_pred_stress.append(Truth_history_stress[i_array,:,:])

        ground_truth_face.append(face_history[i_array,:,:])
        ground_pred_face.append(face_history[i_array,:,:])

    for i_arr in range(Net_pred.shape[0]):
        ground_truth.append(Truth_pred[i_arr,:,:])
        ground_pred.append(Net_pred[i_arr,:,:])

        ground_truth_stress.append(Truth_pred_stress[i_arr,:,:])
        ground_pred.append(Net_pred_stress[i_arr,:,:])

        ground_truth_face.append(face_ground_truth[i_arr,:,:])
        ground_pred_face.append(face_ground_pred[i_arr,:,:])

    time_step = len(ground_truth)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(18, 6))
   
    # 创建一个图形
    # 创建一个包含两个子图的绘图窗口    
    # ax1.clear()
    # ax2.clear()
    pos = ground_truth[0]
    pos1 = scale_factor*(pos[:,3:6] - pos[:,0:3]) + pos[:,0:3]
    face = ground_truth_face[0]
    stress1 = ground_truth_stress[0]
    face1 = np.concatenate((face[:,[0,1,2]],face[:,[0,1,3]],face[:,[0,2,3]],face[:,[1,2,3]]),axis = 0)
    face1 = np.sort(face1,axis = -1)
    face1= np.unique(face1,axis = 0)    
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([0, 40])
    ax1.set_zlim([-10, 10])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_aspect('equal')
    ax1.view_init(30, 40)
    ax1.plot_trisurf(pos1[:, 0], pos1[:, 1],face1, pos1[:, 2], shade=True, linewidth = 0.5, edgecolor = 'grey')    
    ax1.set_title('Surface Plot of Ground Trurh')
   
    
    pos = ground_pred[0]
    pos2 = scale_factor*(pos[:,3:6] - pos[:,0:3]) + pos[:,0:3]
    face = ground_truth_face[0]
    stress2 = ground_truth_stress[0]
    face2 = np.concatenate((face[:,[0,1,2]],face[:,[0,1,3]],face[:,[0,2,3]],face[:,[1,2,3]]),axis = 0)
    face2 = np.sort(face1,axis = -1)
    face2= np.unique(face1,axis = 0)
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([0, 40])
    ax2.set_zlim([-10, 10])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(30, 40)
    ax2.set_aspect('equal')
    ax2.plot_trisurf(pos2[:, 0], pos2[:, 1],face2, pos2[:, 2], shade=True, linewidth = 0.5, edgecolor = 'grey', color="cyan")   
    ax2.set_title('Surface Plot of RLGT')
    # fig.suptitle('The trajectory of the paper at step {}'.format(frame), fontsize=16)
    plt.show()
    ani = FuncAnimation(fig, update, frames=np.arange(time_step),interval=5)    
    filename_save  = os.path.join(root_path,'paper_traj_{:06d}.gif'.format(i_test))  
    # 保存动画为MP4文件
    ani.save(filename_save, writer='pillow', fps=30)
    plt.close('all')

    