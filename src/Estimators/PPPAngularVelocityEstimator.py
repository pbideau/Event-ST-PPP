from Estimators.PPPVelocityEstimator import PPPVelocityEstimator
from utils.utils import *


class PPPAngularVelocityEstimator(PPPVelocityEstimator):
    def __init__(self, dataset, dataset_path, sequence, Ne, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250
                    ) -> None:
        super().__init__(   dataset, 
                            dataset_path, 
                            sequence, 
                            Ne, 
                            overlap, 
                            fixed_size, 
                            padding, 
                            optimizer, 
                            optim_kwargs, 
                            lr, 
                            lr_step, 
                            lr_decay, 
                            iters)
        self.trans_type = "rot"
