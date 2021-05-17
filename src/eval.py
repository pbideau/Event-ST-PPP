#%%
from Evaluator.AngularEvaluator import AngularEvaluator
from Evaluator.LinearEvaluator import  LinearEvaluator
import argparse
from utils.utils import *

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo code to extract motion from eventdata')
    parser.add_argument('-t', '--transformation', default='rot', help="The type of transformation, can be selected from ['rot', 'trans'], default as 'rot'")
    parser.add_argument('-d', '--dataset', default='DAVIS_240C', help="Name of dataset, default as 'DAVIS_240C'")
    args = parser.parse_args()
    trans_type = args.transformation
    dataset = args.dataset
    print("Please select the estimated result file, which should be in the format of [index, reference time, end time, [loss], wx, wy, wz] ([] is optional)")
    filename = selectFilename("Please select the estimated result file")
    if not filename:
        exit()
        
    if trans_type == 'rot':
        print("Angular velocity Evaluator is now used.")
        evaluator = AngularEvaluator(dataset, filename)
    elif trans_type == 'trans':
        print("Linear velocity Evaluator is now used.")
        evaluator = LinearEvaluator(dataset, filename)
    else:
        print("Not supported, please read help!")
        exit()
    evaluator(save_res=False)

# %%

# %%
