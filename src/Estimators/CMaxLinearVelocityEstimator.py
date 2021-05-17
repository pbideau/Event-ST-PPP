from Estimators.CMaxVelocityEstimator import CMaxVelocityEstimator
from utils.utils import *


class CMaxLinearVelocityEstimator(CMaxVelocityEstimator):
    def __init__(self, dataset, dataset_path, sequence, Ne, overlap=0, N=1, fixed_size=False, padding=0, t_start=0, t_stop=60,
                    optimizer='Adam', optim_kwargs=None, lr=0.01, lr_step=80, lr_decay=0.1, iters=80
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
        self.trans_type = "trans"

    def warp_event(self, poses, events):
        linear_vel_vector = self.linear_velocity_vector(poses)
        point_3d = self.events_form_3d_points(events)
        dt = events[:, 0].unsqueeze(1)
        coordinate_3d = point_3d - dt * linear_vel_vector
        warped_x, warped_y = self.events_back_project(coordinate_3d)
        warped_events = torch.stack((dt.squeeze(), warped_x, warped_y, events[:, 3]), dim=1)
        return warped_events.squeeze()