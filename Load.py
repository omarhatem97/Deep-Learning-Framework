import os
import numpy as np

"""
1- path_of_file 
load image from folder 
Args:
    path_folder : path of folder of dataset
Returns:
    array of paths of files 
"""
def path_of_file (path_folder):
    folder = path_folder
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    return onlyfiles
