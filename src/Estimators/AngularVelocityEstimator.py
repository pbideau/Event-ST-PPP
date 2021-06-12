from Estimators.VelocityEstimator import VelocityEstimator
from utils.utils import *
        
class AngularVelocityEstimator(VelocityEstimator):
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
    
    def warp_event(self, poses, events):
        angular_vel_matrix = self.angular_velocity_matrix(poses)
        point_3d = self.events_form_3d_points(events)
        dt = events[:, 0]
        point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)
        r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)
        coordinate_3d = point_3d - r
        
        warped_x, warped_y = self.events_back_project(coordinate_3d)
        warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)
        return warped_events.squeeze()