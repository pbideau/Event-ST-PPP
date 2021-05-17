from Estimators.EstimatorFactory import EstimatorFactory
import os
from utils.utils import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Demo code to extract motion from eventdata')
    parser.add_argument('-p', '--path', default='dataset', help="Path to dataset")
    parser.add_argument('-d', '--dataset', default='DAVIS_240C', help="Name of dataset, default as 'DAVIS_240C'")
    parser.add_argument('-s', '--seq', default='dynamic_rotation', help="Name of sequence, default as 'dynamic_rotation'")
    parser.add_argument('-o', '--output', default='output.txt', help="Name for output file, default as 'output.txt'")
    parser.add_argument('-n', '--Ne', default=30000, help="The number of events per batch, default as 30000")
    parser.add_argument('-a', '--alpha', default=0.1, help="'alpha' of gamma prior, default as 0.1")
    parser.add_argument('-b', '--beta', default=1.59, help="'beta' of gamma prior, default as 1.59")
    parser.add_argument('-l', '--lr', default=0.05, help="Learning rate of optimization, defualt as 0.05")
    parser.add_argument('-i', '--iter', default=250, help="Maximum number of iterations, default as 250")
    parser.add_argument('-m', '--method', default='st-ppp', help="The name of method, can be selected from ['st-ppp', 'cmax'], default as 'st-ppp'")
    parser.add_argument('-t', '--transformation', default='rot', help="The type of transformation, can be selected from ['rot', 'trans'], default as 'rot'")
    parser.add_argument('-f', '--figure', action='store_true', default=False, help="Save figures or not, default as False, use '-f' to set the flag")


    args = parser.parse_args()
    dataset_path = args.path
    dataset = args.dataset
    sequence = args.seq
    output_filename = args.output
    Ne = int(args.Ne)
    alpha = float(args.alpha)
    beta = float(args.beta)
    lr = float(args.lr)
    iters = int(args.iter)
    method = args.method
    trans_type = args.transformation
    save_figs = args.figure

    res_save_dir = os.path.join('output/', sequence)
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    save_filepath = os.path.join(res_save_dir, output_filename)

    VE = EstimatorFactory(  method = method, 
                            transformation = trans_type, 
                            dataset=dataset, 
                            dataset_path=dataset_path, 
                            sequence=sequence, 
                            Ne=Ne, 
                            overlap=0, 
                            fixed_size=True, 
                            padding=100, 
                            lr=lr, 
                            lr_step=iters, 
                            iters=iters).get_estimator()

    # set save_figs to be False while not saving output figures
    VE(save_filepath, alpha, beta, count=1, save_figs=save_figs)
