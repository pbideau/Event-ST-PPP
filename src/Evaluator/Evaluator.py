#%%
from utils.utils import rad2degree
import numpy as np
from scipy.interpolate import interp1d
from visualize.visualize import compare_plot_func
from utils.utils import *
import json

class Evaluator(object):
    def __init__(self, dataset, res_dir) -> None:
        super().__init__()
        self.dataset = dataset
        self.gt = np.array([])
        self.es = np.array([])
        self.get_gt()
        self.get_es(res_dir)
        self.res = {}

    def __call__(self, save_res = False) -> None:
        self.simple_evaluate()
        if save_res:
            save_dir = selectDirectory()
            jobject = json.dumps(self.res)
            with open('{}/eval_res.json'.format(save_dir),'w') as f:
                f.write(jobject)
    
    def get_gt(self):
        name = self.dataset
        print("Please select the ground-truth file, for DAVIS_240C dataset is imu.txt.")
        gt_filename = selectFilename("Please select the ground-truth file, e.g. imu.txt for DAVIS_240C.")
        if not gt_filename:
            exit()
        cam_vel = []
        if name == 'DAVIS_240C':
            imu_data = np.loadtxt(gt_filename)
            for cam_info in imu_data:
                temp = np.array(
                    [cam_info[0] - 0.0024, cam_info[4], cam_info[5], cam_info[6]])
                cam_vel.append(temp)
        else:
            print("Not supported!")

        print(len(cam_vel))
        self.gt = np.array(cam_vel)
        self.gt_max = np.max(self.gt[:,1:])
        self.gt_min = np.min(self.gt[:,1:])
    
    def get_es(self, output_dir):
        self.es = np.loadtxt(output_dir)

    def data_selection(self, data1, data2, intp = True):
        i = 0
        j = 0
        dst = []

        if intp == True:
            dst = data2[:,0]
            for i in range(data1.shape[1]-1):
                x = data1[:,0]
                y = data1[:,i+1]
                f = interp1d(x, y, kind = 'linear')
                new_y = f(data2[:,0])
                dst = np.c_[dst, new_y]
            return np.array(dst)
        
        else:
            while i < len(data1) and j < len(data2):
                p = data1[i]
                q = data2[j]
                i += 1
                if p[0] < q[0]:
                    continue
                else:
                    dst.append(p)
                    j += 1
            return np.array(dst)

    def rms(self, data1, data2):
        return np.sqrt(np.mean(np.power(data1-data2, 2)))

    def rms_r(self, data1,data2):
        return np.sqrt(np.mean(np.power(data1-data2, 2)))/(self.gt_max-self.gt_min)

    def std(self, data1, data2):
        return np.std(data1-data2)

    def std_r(self, data1, data2):
        return np.std(data1-data2)/(np.max(data1)-np.min(data1))

    def max_err(self, data1, data2):
        return np.max(np.abs(data1-data2))

    def max_err_r(self, data1, data2):
        return np.max(np.abs(data1-data2))/(np.max(data1)-np.min(data1))

    def mean_err(self, data1, data2):
        return np.mean(np.abs(data1-data2))
    
    def mean_err_r(self, data1, data2):
        return np.mean(np.abs(data1-data2))/(np.max(data1)-np.min(data1))

    def save_res_as_dict(self,**kwargs):
        self.res['rms'] = kwargs['rms']
        self.res['rms%'] = kwargs['rmsp']
        self.res['mean'] = kwargs['mean']
        self.res['mean%'] = kwargs['meanp']
        self.res['std'] = kwargs['std']
        self.res['std%'] = kwargs['stdp']
        self.res['mean_x'] = kwargs['meanx']
        self.res['mean_y'] = kwargs['meany']
        self.res['mean_z'] = kwargs['meanz']

    def simple_pre(self, save=True):
        gt = self.gt
        es = self.es
        estimated_vel = es[:,-3:]
        # estimated_vel = np.array(list(map(degree2rad, estimated_vel)))
        estimated_ts = (es[:, 1]+es[:, 2])/2
        es = np.c_[estimated_ts, estimated_vel]
        compare_plot_func(gt, es, 3, 0, 60, save, mark ='--')
        selected_data = self.data_selection(gt, es)
        return selected_data, es
    
    
    def simple_evaluate(self):
        gt, es = self.simple_pre()

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

        print('rms:{}\nrms%:{}\nmax error:{}\nmax error%:{}\nmean error:{}\nmean error of x:{}\nmean error of y:{}\nmean error of z:{}\nstd:{}'.format(rad2degree(_rms), _rms_r, rad2degree(max_e), max_e_r, 
            rad2degree(mean_e), rad2degree(mean_err_x), rad2degree(mean_err_y), rad2degree(mean_err_z),rad2degree(_std)))
        
        self.save_res_as_dict(rms = rad2degree(_rms), rmsp = _rms_r, mean = rad2degree(mean_e), meanp = mean_e_r, meanx = rad2degree(mean_err_x), meany = rad2degree(mean_err_y), meanz = rad2degree(mean_err_z), std = rad2degree(_std), stdp = _std_r)
