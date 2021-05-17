# Demo Code for Event Alignment

This is the implementation of the submission "The Spatio-Temporal Poisson Point Process: A Simple model for the Alignment of Event data" (ID 6649) in Python 3 . The code takes an event sequence recorded by an event camera as input and accurately registers events to each other. In addition the code estimates the camera's angular- and linear velocity.

<p align="center">
  <img height="300" src="/imgs/overview-diagram.png">
</p>


## Requirements
    pandas==1.0.5
    matplotlib==3.2.1
    numpy==1.18.5
    opencv_python==4.2.0.34
    torch==1.8.0
    scikit_learn==0.24.2
    scipy==1.4.1

## Getting Started

A demo version of the code can be run by simply executing the command:
    
    python src/main.py

It will run with default arguments (as used in the experiments of the submitted paper).
The output is (1) an estimate of the angular velocity of the event camera and (2) aligned events that can be visualized via an image of events showing sharp object contours. Both outputs can be found in the output folder.

The record in the output.txt is in the format of:

[index, reference time, end time, loss value, &omega;<sub>x</sub>, &omega;<sub>y</sub>, &omega;<sub>z</sub> ]

### Angular velocity estimation

The provided demo version runs the code on a short subsequence of the event sequence *dynamic_rotatation* of the *DAVIS_240C dataset* estimating the angular velocity of the event camera. It can be also run by executing:

    python src/main.py --seq dynamic_rotation -t 'rot'


### Linear velocity estimation

Optionally the code is designed to also estimate the linear velocity of an event camera. This can be run on a short example sequence by executing:

    python src/main.py --seq dynamic_translation -t 'trans'
    
### Optional parameter settings

Details regarding optional parameters settings of the code are shown below:

    usage: main.py [-h] [-p PATH] [-d DATASET] [-s SEQ] [-o OUTPUT] [-n NE] [-a ALPHA] [-b BETA] [-l LR] [-i ITER] [-m METHOD] [-t TRANSFORMATION] [-f]

    Code to extract motion from event-data

    optional arguments:
    -h, --help            show this help message and exit
    -p PATH, --path PATH  Path to dataset
    -d DATASET, --dataset DATASET
                            Name of dataset, default as 'DAVIS_240C'
    -s SEQ, --seq SEQ     Name of sequence, default as 'dynamic_rotation'
    -o OUTPUT, --output OUTPUT
                            Name for output file, default as 'output.txt'
    -n NE, --Ne NE        The number of events per batch, default as 30000
    -a ALPHA, --alpha ALPHA
                            'alpha' of gamma prior, default as 0.1
    -b BETA, --beta BETA  'beta' of gamma prior, default as 1.59
    -l LR, --lr LR        Learning rate of optimization, defualt as 0.05
    -i ITER, --iter ITER  Maximum number of iterations, default as 250
    -m METHOD, --method METHOD
                            The name of method, can be selected from ['st-ppp', 'cmax'], default as 'st-ppp'
    -t TRANSFORMATION, --transformation TRANSFORMATION
                            The type of transformation, can be selected from ['rot', 'trans'], default as 'rot'
    -f, --figure          Save figures or not, default as False, use '-f' to set the flag

## Evaluation

The code also provides evaluation part. Mainly two files are necessary for the evaluation, the result file from estimation and the groundtruth file.

Please notice that the result file need to be in a certain format:

    index, reference time, end time, [loss], wx, wy, wz ([*] is optional)

### How to run it?

Simply run the code using command:

    python src/eval.py

Some dialog boxes for choosing files would come out. Please follow the instructions shown in the command line window or the title of dialog box.




