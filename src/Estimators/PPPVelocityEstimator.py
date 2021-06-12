from Estimators.VelocityEstimator import VelocityEstimator
from utils.utils import *


class PPPVelocityEstimator(VelocityEstimator):
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
        self.method = "st-ppp"
        
    def loss_func(self, x, events_batch, *args) -> torch.float32:
        warped_events_batch = self.warp_event(x, events_batch) 
        frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
        frame = convGaussianFilter(frame)
        loss,_ = self.poisson(frame.abs(), *args)
        return loss
    
    def calResult(self, events_batch, para, *args,  warp = True, cal_loss = True, fixed_size = False, padding = 0):
        device = events_batch.device
        poses = torch.from_numpy(para).float().to(device)
        with torch.no_grad():
            if warp:
                warped_events_batch = self.warp_event(poses, events_batch)
            else:
                point_3d = self.events_form_3d_points(events_batch)
                warped_x, warped_y = self.events_back_project(point_3d)
                warped_events_batch = torch.stack((events_batch[:, 0], warped_x, warped_y, events_batch[:, 3]), dim=1)

            frame = self.events2frame(warped_events_batch, fixed_size = fixed_size, padding = padding)
            frame = convGaussianFilter(frame)
            img = frame.sum(axis=0).cpu().detach().numpy()

            if cal_loss:
                loss, map = self.poisson(frame.abs(), *args)
                loss = loss.item()
                map = map.cpu().detach().numpy()

            else:
                loss = 0
                map = 0

            torch.cuda.empty_cache()

        return frame, loss, img, map