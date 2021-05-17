import matplotlib.pyplot as plt
import os
import numpy as np

def plot_img_map(img, map, clim = 4, cb_max = 8, filepath="output", save = False):
    img_0, img_1 = img
    map_0, map_1 = map
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1) 
    plt.title('aligned events')
    plt.imshow(img_1, cmap='bwr')
    plt.clim(-clim,clim)
    plt.colorbar()

    plt.subplot(2, 2, 2) 
    plt.title('unaligned events')
    plt.imshow(img_0, cmap='bwr')
    plt.clim(-clim,clim)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('-log(likelihood) map of aligned events')
    plt.imshow(map_1, cmap='jet')
    plt.clim(0,cb_max)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('-log(likelihood) map of unaligned events')
    plt.imshow(map_0, cmap='jet')
    plt.clim(0,cb_max)
    plt.colorbar()

    if save:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, 'output.png')
        plt.savefig(filepath)

def compare_plot_func(data_groundtruth, data_estimated, dof, t_start, t_stop, save, mark='m^'):
    if dof == 3:
        ts_gt = data_groundtruth[:, 0]
        pitch_gt = np.rad2deg(data_groundtruth[:, 1])
        yaw_gt = np.rad2deg(data_groundtruth[:, 2])
        roll_gt = np.rad2deg(data_groundtruth[:, 3])

        index_start = np.min(np.where(ts_gt >= t_start))
        index_stop = np.max(np.where(ts_gt <= t_stop))
        ts_gt = ts_gt[index_start:index_stop]
        pitch_gt = pitch_gt[index_start:index_stop]
        yaw_gt = yaw_gt[index_start:index_stop]
        roll_gt = roll_gt[index_start:index_stop]

        ts_e = data_estimated[:, 0]
        pitch_e = np.rad2deg(data_estimated[:, 1])
        yaw_e = np.rad2deg(data_estimated[:, 2])
        roll_e = np.rad2deg(data_estimated[:, 3])

        index_start = np.min(np.where(ts_e >= t_start))
        index_stop = np.max(np.where(ts_e <= t_stop))
        ts_e = ts_e[index_start:index_stop]
        pitch_e = pitch_e[index_start:index_stop]
        yaw_e = yaw_e[index_start:index_stop]
        roll_e = roll_e[index_start:index_stop]

        plt.figure(figsize=(30, 18))

        plt.subplot(3, 1, 1)
        plt.plot(ts_gt, pitch_gt)
        plt.plot(ts_e, pitch_e, mark, linewidth=1)
        plt.grid()

        plt.legend(('ground truth', 'estimated'), loc='upper right')
        plt.title('pitch')
        plt.xlabel('t[s]')
        plt.ylabel('velocity[radian/s]')

        plt.subplot(3, 1, 2)
        plt.plot(ts_gt, yaw_gt)
        plt.plot(ts_e, yaw_e, mark, linewidth=1)
        plt.grid()

        plt.legend(('ground truth', 'estimated'), loc='upper right')
        plt.title('yaw')
        plt.xlabel('t[s]')
        plt.ylabel('velocity[radian/s]')
        plt.subplot(3, 1, 3)
        plt.plot(ts_gt, roll_gt)
        plt.plot(ts_e, roll_e, mark, linewidth=1)
        plt.grid()

        plt.legend(('ground truth', 'estimated'), loc='upper right')
        plt.title('roll')
        plt.xlabel('t[s]')
        plt.ylabel('velocity[radian/s]')
        if save:
            plt.savefig('estimated_pyr.png', transparent=False)
    
    elif dof == 2:
        ts_gt = data_groundtruth[:, 0]
        x_gt = data_groundtruth[:, 1]
        y_gt = data_groundtruth[:, 2]
        z_gt = data_groundtruth[:, 3]

        index_start = np.min(np.where(ts_gt >= t_start))
        index_stop = np.max(np.where(ts_gt <= t_stop))
        ts_gt = ts_gt[index_start:index_stop]
        x_gt = x_gt[index_start:index_stop]
        y_gt = y_gt[index_start:index_stop]
        z_gt = z_gt[index_start:index_stop]

        ts_e = data_estimated[:, 0]
        x_e = data_estimated[:, 1]
        y_e = data_estimated[:, 2]
        z_e = data_estimated[:, 3]

        index_start = np.min(np.where(ts_e >= t_start))
        index_stop = np.max(np.where(ts_e <= t_stop))
        ts_e = ts_e[index_start:index_stop]
        x_e = x_e[index_start:index_stop]
        y_e = y_e[index_start:index_stop]
        z_e = z_e[index_start:index_stop]

        plt.figure(figsize=(30, 18))

        plt.subplot(3, 1, 1)
        plt.plot(ts_gt, x_gt)
        plt.plot(ts_e, x_e, mark, linewidth=1)
        plt.grid()
        plt.legend(('ground truth', 'estimated'), loc='upper right')
        plt.title('x')
        plt.xlabel('t[s]')
        plt.ylabel('velocity')

        plt.subplot(3, 1, 2)
        plt.plot(ts_gt, y_gt)
        plt.plot(ts_e, y_e, mark, linewidth=1)
        plt.grid()
        plt.legend(('ground truth', 'estimated'), loc='upper right')
        plt.title('y')
        plt.xlabel('t[s]')
        plt.ylabel('velocity')

        plt.subplot(3, 1, 3)
        plt.plot(ts_gt, z_gt)
        plt.plot(ts_e, z_e, mark, linewidth=1)
        plt.grid()
        plt.legend(('ground truth', 'estimated'), loc='upper right')
        plt.title('z')
        plt.xlabel('t[s]')
        plt.ylabel('velocity')
        if save:
            plt.savefig('estimated_xy.png', transparent=False)
    else:
        print("'dof' can only be 2 or 3, 3 refers to rotation, 2 refers to translation.")
