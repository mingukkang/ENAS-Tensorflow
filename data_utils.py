import os
import sys
import random
import cv2
import numpy as np
from glob import *
import utils
from utils import plot_data_label

random.seed(random.randint(0, 2 ** 31 - 1))

def _read_data(data_path, channel, img_size, n_aug_img):

    ccc = channel
    global ccc

    if n_aug_img == 1:
        aug_flag = False
    else:
        aug_flag = True

    class_list = os.listdir(data_path)
    class_list.sort()
    n_classes = len(class_list)
    images= []
    image_holder = []
    labels= []
    label_holder = []

    for i in range(n_classes):
        img_class = glob(os.path.join(data_path,class_list[i]) + '/*.*')
        images += img_class
        for j in range(len(img_class)):
            labels += [i]

    if channel == "1":
        flags = 0
    else:
        flags = 1

    length_data = len(images)
    for j in range(length_data):
        img = cv2.imread(images[j],flags = flags)

        if channel ==1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(img_size,img_size))
            img = img.astype(np.float32)

            # Augmentation
            if aug_flag is True:
                for k in range(n_aug_img - 1):
                    image_aug = img_augmentation(img)
                    image_aug = np.reshape(image_aug, [1,img_size, img_size, channel])
                    image_holder.append(image_aug)
                    label_holder.append(labels[j])

            img = np.reshape(img, [1, img_size, img_size, channel])

            image_holder.append(img)
            label_holder.append(labels[j])

        else:
            img = cv2.resize(img,(img_size,img_size))
            img = img.astype(np.float32)

            if aug_flag is True:
                for k in range(n_aug_img - 1):
                    image_aug = img_augmentation(img)
                    image_aug = np.reshape(image_aug, [1,img_size, img_size, channel])
                    image_holder.append(image_aug)
                    label_holder.append(labels[j])

            img = np.reshape(img, [1,img_size,img_size,channel])

            image_holder.append(img)
            label_holder.append(labels[j])

    image_holder =np.concatenate(image_holder, axis = 0)
    label_holder = np.asarray(label_holder, dtype = np.int32)

    if aug_flag is True:
        images = []
        labels = []

        for w in range(n_aug_img):
            holder = []
            total_data = length_data*n_aug_img
            quota = (length_data//n_classes)
            interval = total_data//n_classes
            for r in range(n_classes):
                temp = np.add(np.full((quota), interval*r + quota*w),np.random.permutation(quota))
                holder.extend(temp)

            _ = random.shuffle(holder)
            images.append(image_holder[holder].astype(np.float32))
            labels.append(label_holder[holder])

        images = np.concatenate(images, axis = 0)
        labels = np.concatenate(labels, axis = 0)

    else:
        idx = np.random.permutation(length_data)
        images = image_holder[idx]
        labels = label_holder[idx]

    n_batch_mean = len(images)
    mean = 0
    std = 0
    for b in range(n_batch_mean):
        mean += np.mean(images[b], axis = (0,1,2))/n_batch_mean
        std += np.std(images[b], axis = (0,1,2))/n_batch_mean
    plot_data_label(images[0:64]/255, labels[0:64],channel ,8,8,8)
    print(data_path)
    print("Mean:", mean)
    print("Std:", std)
    print("_____________________________")
    images = (images - mean) / std

    return images, labels

def read_data(train_dir,val_dir, test_dir, channel, img_size, n_aug_img):
    '''
    channel = channels of images. MNIST: channels = 1
                                Cifar 10: channels = 3
    '''
    print("-"*80)
    print("Reading data")

    images, labels = {}, {}

    images["train"], labels["train"] = _read_data(train_dir, channel, img_size,n_aug_img)
    images["valid"], labels["valid"] = _read_data(val_dir, channel, img_size, 1)
    images["test"], labels["test"] = _read_data(test_dir, channel, img_size, 1)

    return images, labels

def img_augmentation(image):
    global ccc

    def Flip(image):
        img_filped = cv2.flip(image, 1)

        return img_filped

    def enlarge(image, magnification):

        H_before = np.shape(image)[1]
        center = H_before // 2
        M = cv2.getRotationMatrix2D((center, center), 0, magnification)
        img_croped = cv2.warpAffine(image, M, (H_before, H_before))

        return img_croped

    def rotation(image):
        H_before = np.shape(image)[1]
        center = H_before // 2
        angle = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((center, center), angle, 1)
        img_rotated = cv2.warpAffine(image, M, (H_before, H_before))

        return img_rotated

    def random_bright_contrast(image):
        alpha = random.uniform(1, 0.1)  # for contrast
        alpha = np.minimum(alpha, 1.3)
        alpha = np.maximum(alpha, 0.7)
        beta = random.uniform(32, 6)  # for brightness

        # g(i,j) = alpha*f(i,j) + beta
        img_b_c = cv2.multiply(image, np.array([alpha]))
        img_b_c = cv2.add(img_b_c, beta)

        return img_b_c

    def aug(image, idx):
        augmentation_dic = {0: enlarge(image, 1.2),
                            1: rotation(image),
                            2: random_bright_contrast(image),
                            3: Flip(image)}

        image = augmentation_dic[idx]
        return image

    if ccc == 3:  # c is number of channel
        l = 4
    else:
        l = 3

    p = [random.random() for m in range(l)]  # 4 is number of augmentation operation
    for n in range(l):
        if p[n] > 0.50:
            image = aug(image, n)

    return image




