import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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


def loadData():
    datasets = pd.read_csv("train.csv")

    #datasets.head() #show first 4 rows
    #datasets.describe() #Return all the statics for every row x col

    #visyalize the data
    y = datasets['label'].iloc[0:10].values
    x = datasets['pixel10'].iloc[0:10].values
    #print(x)
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(x,y,lw=4)
    ax.set_xlabel('pixels',fontsize=14)
    ax.set_ylabel('Its Label',fontsize=14)
    ax.set_title('VisualizeMyData',fontsize=14)

    #plt.show()

    all_pixels = np.array(datasets.iloc[:,1:])
    Labels = np.array(datasets['label'])

    return all_pixels, Labels
