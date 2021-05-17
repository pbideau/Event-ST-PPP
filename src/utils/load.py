import numpy as np
import pandas as pd
import cv2

def undo_distortion(src, instrinsic_matrix, distco=None):
    dst = cv2.undistortPoints(src, instrinsic_matrix, distco, None, instrinsic_matrix)
    return dst
    
def load_dataset(name, path_dataset, sequence):
    if name == "DAVIS_240C":
        calib_data = np.loadtxt('{}/{}/calib.txt'.format(path_dataset,sequence))
        events = pd.read_csv(
            '{}/{}/events.txt'.format(path_dataset,sequence), sep=" ", header=None)
        events.columns = ["ts", "x", "y", "p"]

        fx = calib_data[0]
        fy = calib_data[1]
        px = calib_data[2]
        py = calib_data[3]
        dist_co = calib_data[4:]
        height = 180
        width = 240
        instrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

        LUT = np.zeros([width, height, 2])
        for i in range(width):
            for j in range(height):
                LUT[i][j] = np.array([i, j])
        LUT = LUT.reshape((-1, 1, 2))
        LUT = undo_distortion(LUT, instrinsic_matrix, dist_co).reshape((width, height, 2))
        events_set = events.to_numpy()
    print("Events total count: ", len(events_set))
    print("Time duration of the sequence: {} s".format(events_set[-1][0] - events_set[0][0]))
    return LUT, events_set, height, width, fx, fy, px, py