import numpy as np
import time
from math import inf, nan
from copy import deepcopy
import os
from Estimators.Estimator import Estimator
from visualize.visualize import plot_img_map 
from utils.load import load_dataset
import torch
from torch import optim
from utils.utils import *

        
class VelocityEstimator(Estimator):
    def __init__(self, dataset, dataset_path, sequence, Ne, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250
                    ) -> None:
        LUT, events_set, height, width, fx, fy, px, py = load_dataset(dataset, dataset_path, sequence)
        super().__init__(height, width, fx, fy, px, py, events_set, LUT)
        self.Ne = Ne
        self.sequence = sequence
        self.overlap = overlap
        self.fixed_size = fixed_size
        self.padding = padding
        self.estimated_val = []
        self.img = []
        self.map = []
        self.time_record = []
        self.count = 1
        self.optimizer_name = optimizer
        self.optim_kwargs = optim_kwargs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.iters = iters
        print("Sequence: {}".format(sequence))

    @timer
    def __call__(self, save_filepath, *args, count = 1, save_figs = True, use_prev = True) -> None:
        Ne = self.Ne
        overlap = self.overlap
        para0 = np.array([0, 0, 0])
        self.count = count

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
        self.LUT = torch.from_numpy(self.LUT).float().to(device)

        while True:
            start_time = time.time()

            if overlap:
                events_batch = deepcopy(self.events_set[int(Ne * (self.count - 1) * overlap): int(Ne + Ne * (self.count - 1) * overlap)])
            else:
                events_batch = deepcopy(self.events_set[Ne * (self.count - 1): Ne * self.count])

            if len(events_batch) < Ne:
                break

            t_ref = events_batch[0][0]
            t_end = events_batch[-1][0]

            events_tensor = torch.from_numpy(events_batch).float().to(device)

            events_tensor = undistortion(events_tensor, self.LUT, t_ref)
            print('{}: {}'.format(self.count, t_ref))

            res, loss = self.optimization(para0,  events_tensor, device, *args)
            self.estimated_val.append(np.append([self.count, t_ref, t_end, loss], res))

            # update new initial guess for next frame
            if use_prev:
                para0 = res
            if save_figs:
                img_path = os.path.join('output', self.sequence)
                _, _, img_0, map_0 = self.calResult(events_tensor, np.array([0. ,0. ,0.]), *args, warp = False, fixed_size = False, padding = 50)
                _, _, img_1, map_1 = self.calResult(events_tensor, res, *args, warp=True, fixed_size=False, padding = 50)
                clim = 4 if 'shapes' not in self.sequence else 10
                cb_max = 8 if 'shapes' not in self.sequence else 20
                plot_img_map([img_0, img_1],[map_0, map_1], clim, cb_max, filepath = img_path, save=True)

            self.count += 1
            duration = time.time() - start_time
            print("Duration:{}s\n".format(duration))
            np.savetxt(save_filepath, np.array(self.estimated_val), fmt=[
                '%d', '%.9f', '%.9f', '%.9f', '%.9f', '%.9f', '%.9f'], delimiter=' ')

    def loss_func(self, x, events_batch, *args) -> torch.float32:
        warped_events_batch = self.warp_event(x, events_batch) 
        frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
        frame = convGaussianFilter(frame)
        loss,_ = self.poisson(frame.abs(), *args)

        return loss

    def optimization(self, init_poses, events_tensor, device, *args):
        # initializing local variables for class atrributes
        optimizer_name = self.optimizer_name
        optim_kwargs = self.optim_kwargs
        lr = self.lr
        lr_step = self.lr_step
        lr_decay = self.lr_decay
        iters = self.iters
        if not optim_kwargs:
            optim_kwargs = dict()
        if lr_step <= 0:
            lr_step = max(1, iters)
        
        # preparing data and prameters to be trained
        if init_poses is None:
            init_poses = np.zeros(3, dtype=np.float32)
        
        poses = torch.from_numpy(init_poses.copy()).float().to(device)
        poses.requires_grad = True

        # initializing optimizer
        optimizer = optim.__dict__[optimizer_name](
            [poses],lr =lr, **optim_kwargs)
        scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
        print_interval = 10
        min_loss = inf
        best_poses = poses
        best_it = 0
        # optimization process
        if optimizer_name == 'Adam':
            for it in range(iters):
                optimizer.zero_grad()
                poses_val = poses.cpu().detach().numpy()
                if nan in poses_val:
                    print("nan in the estimated values, something wrong takes place, please check!")
                    exit()
                loss = self.loss_func(poses, events_tensor, *args)
                if it == 0:
                    print('[Initial]\tloss: {:.12f}\tposes: {}'.format(loss.item(), poses_val))
                elif (it + 1) % print_interval == 0:
                    print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}'.format(it + 1, iters, loss.item(), poses_val))
                if loss < min_loss:
                    best_poses = poses
                    min_loss = loss.item()
                    best_it = it
                try:
                    loss.backward()
                except Exception as e:
                    print(e)
                    return poses_val, loss.item()
                optimizer.step()
                scheduler.step()
        else:
            print("The optimizer is not supported.")

        best_poses = best_poses.cpu().detach().numpy()
        print('[Final Result]\tloss: {:.12f}\tposes: {} @ {}'.format(min_loss, best_poses, best_it))
        if device == torch.device('cuda:0'):
            torch.cuda.empty_cache()
        return best_poses, min_loss




