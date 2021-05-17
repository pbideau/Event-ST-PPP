#%%
import numpy as np
from math import *
from visualize.visualize import compare_plot_func
from utils.utils import *
from Evaluator.Evaluator import Evaluator
from sklearn.linear_model import LinearRegression

class LinearEvaluator(Evaluator):
    def __init__(self, dataset, res_dir) -> None:
        super().__init__(dataset, res_dir)

    def get_gt(self):
        name = self.dataset
        print("Please select the ground-truth file, for DAVIS_240C dataset is groundtruth.txt.")
        gt_filename = selectFilename("Please select the ground-truth file, e.g. groundtruth.txt for DAVIS_240C.")
        if not gt_filename:
            exit()
        cam_vel = []
        if name == 'DAVIS_240C':
            imu_data = np.loadtxt(gt_filename)
            for cam_info in imu_data:
                temp = np.array(
                    [cam_info[0] - 0.0024, cam_info[2], cam_info[3], -cam_info[1]])
                cam_vel.append(temp)
        else:
            print("Not supported!")
        self.gt = np.array(cam_vel)
        
    def dev(self, data):
        x = data[:,0]
        y = data[:,1:]
        dt = x[1:]-x[:-1]
        dt = dt[:,np.newaxis]
        y_dev = (y[1:,:]-y[:-1,:])/dt
        return np.c_[x[1:], y_dev]

    def simple_pre(self, save=True):
        gt = self.gt
        gt_dev = self.dev(gt)
        es = self.es
        estimated_vel = es[:,-3:]
        estimated_ts = (es[:, 1] + es[:, 2]) / 2
        es = np.c_[estimated_ts, estimated_vel]
        selected_data = self.data_selection(gt_dev, es)
        es_lr = es[:,1:]
        gt_lr = selected_data[:,1:]
        mag_es = np.sqrt(np.sum(es_lr**2,axis = 1))[:, np.newaxis]
        mag_gt = np.sqrt(np.sum(gt_lr**2,axis = 1))[:, np.newaxis]
        linear_reg = LinearRegression(fit_intercept=False)
        reg = linear_reg.fit(mag_es, mag_gt)
        print(reg.coef_)
        es = np.c_[es[:,0],es[:,1:]*reg.coef_[0]]
        self.gt_max = np.max(gt_dev[:,1:])
        self.gt_min = np.min(gt_dev[:,1:])
        compare_plot_func(gt_dev, es, 2, 0, 60, save)
        return selected_data, es
    
    def unit_vector_plot(self, data1, data2):
        t1 = data1[:,0]
        x1 = data1[:,1:]
        x1 = x1 - np.mean(x1,axis = 0)
        t2 = data2[:,0]
        x2 = data2[:,1:]
        norm1 = np.sqrt(np.sum(x1**2, axis = 1))
        norm2 = np.sqrt(np.sum(x2**2, axis = 1))
        x1_unit = x1/norm1[:,np.newaxis]
        x2_unit = x2/norm2[:,np.newaxis]
        gt = np.c_[t1,x1_unit]
        es = np.c_[t2,x2_unit]
        compare_plot_func(gt, es, self.sequence, 2, 0, 60, True)
    
    def simple_evaluate(self):
        gt, es = self.simple_pre()
        print(self.gt_max, self.gt_min)
        _rms = self.rms(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))
        _rms_r = self.rms_r(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))

        max_e = self.max_err(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))
        max_e_r = self.max_err_r(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))

        mean_e = self.mean_err(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))
        mean_e_r = self.mean_err_r(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))

        mean_err_x = self.mean_err(gt[:, 1], es[:, 1])
        mean_err_y = self.mean_err(gt[:, 2], es[:, 2])
        mean_err_z = self.mean_err(gt[:, 3], es[:, 3])

        _std = self.std(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))
        _std_r = self.std_r(np.reshape(gt[:, 1:], (-1,)), np.reshape(es[:, 1:], (-1,)))

        print('rms:{}\nrms%:{}\nmax error:{}\nmax error%:{}\nmean error:{}\nmean error of x:{}\nmean error of y:{}\nmean error of z:{}\nstd:{}'.format(_rms, _rms_r, max_e, max_e_r, mean_e, mean_err_x, mean_err_y, mean_err_z, _std))
        
        self.save_res_as_dict(rms = _rms, rmsp = _rms_r, mean = mean_e, meanp = mean_e_r, meanx = mean_err_x, meany = mean_err_y, meanz = mean_err_z, std = _std, stdp = _std_r)
