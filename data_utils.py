import os
import sys
import cv2
import numpy as np
from glob import *

import utils

def _read_data(data_path, channel):
    class_list = os.listdir(data_path)
    class_list.sort()
    n_classes = len(class_list)
    image_list = []
    image_holder = []
    label_list = []

    for i in range(n_classes):
        img_class = glob(os.path.join(data_path,class_list[i]) + '/*.*')
        image_list += img_class
        for j in range(len(img_class)):
            label_list += [i]

    if channel == "1":
        flags = 0
    else:
        flags = 1
    length_data = len(image_list)
    for j in range(length_data):
        img = cv2.imread(image_list[j],flags = flags)
        if channel ==1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(32,32))
            img = img.astype(np.float32)
            img = np.reshape(img,[1,32,32,channel])
        else:
            img = img.resize(img,(32,32))
            img = img.astype(np.float32)
        image_holder.append(img)

    image_holder =np.concatenate(image_holder, axis = 0)
    label_holder = np.asarray(label_list, dtype = np.int32)

    idx = np.random.permutation(length_data)
    images = image_holder[idx]
    labels = label_holder[idx]

    ## preprocessing
    mean = np.mean(images, axis = (0,1,2))
    std = np.std(images, axis = (0,1,2))
    images = (images - mean)/std

    return images, labels

def read_data(train_dir,val_dir, test_dir, channel):
    '''
    channel = channels of images. MNIST: channels = 1
                                Cifar 10: channels = 3
    '''
    print("-"*80)
    print("Reading data")

    images, labels = {}, {}

    images["train"], labels["train"] = _read_data(train_dir, channel)
    images["valid"], labels["valid"] = _read_data(val_dir, channel)
    images["test"], labels["test"] = _read_data(test_dir, channel)

    return images, labels


