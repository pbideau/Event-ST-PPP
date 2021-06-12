from Estimators.PPPAngularVelocityEstimator import PPPAngularVelocityEstimator
from Estimators.PPPLinearVelocityEstimator import PPPLinearVelocityEstimator
from Estimators.CMaxAngularVelocityEstimator import CMaxAngularVelocityEstimator
from Estimators.CMaxLinearVelocityEstimator import CMaxLinearVelocityEstimator

class EstimatorFactory(object):
    def __init__(self, method, transformation, dataset, dataset_path, sequence, Ne, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250
                    ) -> None:
        if method == "st-ppp":
            if transformation == "rot":
                print("Using method: Poisson Point Process for angular velocity estimation")
                self.VE = PPPAngularVelocityEstimator(dataset=dataset, dataset_path=dataset_path, sequence=sequence, Ne=Ne, overlap=0, fixed_size=True, padding=100, lr=lr, lr_step=iters, iters=iters)
            elif transformation == "trans":
                print("Using method: Poisson Point Process for linear velocity estimation")
                self.VE = PPPLinearVelocityEstimator(dataset=dataset, dataset_path=dataset_path, sequence=sequence, Ne=Ne, overlap=0, fixed_size=True, padding=100, lr=lr, lr_step=iters, iters=iters)
            else:
                print("The transformation is not supported, please read help")
                exit()

        elif method == "cmax":
            if transformation == "rot":
                print("Using method: Contrast maximization for angular velocity estimation")
                self.VE = CMaxAngularVelocityEstimator(dataset=dataset, dataset_path=dataset_path, sequence=sequence, Ne=Ne, overlap=0, fixed_size=True, padding=100, lr=lr, lr_step=iters, iters=iters)
            elif transformation == "trans":
                print("Using method: Contrast maximization for linear velocity estimation")
                self.VE = CMaxLinearVelocityEstimator(dataset=dataset, dataset_path=dataset_path, sequence=sequence, Ne=Ne, overlap=0, fixed_size=True, padding=100, lr=lr, lr_step=iters, iters=iters)
            else:
                print("The transformation is not supported, please read help")
                exit()
        
        else:
            print("The method is not supported, please read help")
            exit()

    def get_estimator(self):
        return self.VE