from Estimators.CMaxVelocityEstimator import CMaxVelocityEstimator
from Estimators.AngularVelocityEstimator import AngularVelocityEstimator
from utils.utils import *
        
class CMaxAngularVelocityEstimator(CMaxVelocityEstimator, AngularVelocityEstimator):
    def __init__(self, dataset, dataset_path, sequence, Ne, overlap=0, fixed_size = False, padding = 0,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.01, lr_step = 80, lr_decay = 0.1, iters = 80
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