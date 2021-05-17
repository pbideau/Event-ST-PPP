import numpy as np
import time
import torch
import torch.nn.functional as F
from tkinter import *
import tkinter.filedialog
from Estimators.GaussianSmoothing import GaussianSmoothing

def selectDirectory(title):
    root = tkinter.Tk()
    root.withdraw()
    return tkinter.filedialog.askdirectory(title = title)

def selectFilename(title):
    root = tkinter.Tk()
    root.withdraw()
    return tkinter.filedialog.askopenfilename(title = title)

def selectFilenames():
    root = tkinter.Tk()
    root.withdraw()
    return tkinter.filedialog.askopenfilenames()

def undistortion(events_batch, LUT, t_ref):
    events_batch[:, 0] = events_batch[:, 0] - t_ref
    events_batch[:, 1:3] = LUT[(events_batch[:, 1]).type(torch.long), (events_batch[:, 2]).type(torch.long), :]
    return events_batch

def timer(func):
    def acc_time(self, *args, **kwargs):
        start = time.time()
        func(self, *args, **kwargs)
        end = time.time()
        main_duration = end - start
        print('The estimation lasts for {} h {} m {} s.'.format(int(main_duration/3600),int(main_duration%3600/60),int(main_duration%3600%60)))
    return acc_time

def rad2degree(val):
    return val/np.pi*180.

def degree2rad(val):
    return val/180*np.pi

def convGaussianFilter(frame, k_size = 5, sigma = 1):
    device = frame.device
    dim = frame.size()
    smoothing = GaussianSmoothing(dim[0], k_size, sigma, device)
    frame = torch.unsqueeze(frame, 0)
    frame = F.pad(frame, (2, 2, 2, 2), mode='reflect')
    frame = torch.squeeze(smoothing(frame))
    return frame
