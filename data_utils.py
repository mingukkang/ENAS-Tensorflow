import os
import sys
import random
import cv2
import numpy as np
from glob import *
import utils

def _read_data(data_path, channel, img_size, n_aug_img):

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

            # preprocessing
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            img = (img - mean) / std

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

            # preprocessing
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            img = (img - mean) / std

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

    images = []
    labels = []

    for w in range(n_aug_img):
        idx = np.random.permutation(length_data)
        images.append(image_holder[idx].astype(np.float32))
        labels.append(label_holder[idx])

    images = np.concatenate(images, axis = 0)
    labels = np.concatenate(labels, axis = 0)

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

    def enlarge(image,magnification):

        H_before = np.shape(image)[1]
        center = H_before//2
        M = cv2.getRotationMatrix2D((center,center),0,magnification)
        img_croped = cv2.warpAffine(image,M,(H_before, H_before))

        return img_croped

    def rotation(image):
        H_before = np.shape(image)[1]
        center = H_before // 2
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((center,center),angle,1)
        img_rotated = cv2.warpAffine(image,M,(H_before, H_before))

        return img_rotated

    def random_bright_contrast(image):
        alpha = random.uniform(0.3, 1.0) # for contrast
        beta = random.uniform(0.3, 1.0) # for brightness

        # g(i,j) = alpha*f(i,j) + beta
        img_b_c = cv2.multiply(image, np.array([alpha]))
        img_b_c = cv2.add(img_b_c, beta)

        return img_b_c

    def gaussian_noise(image):
        noise = np.random.normal(size=np.shape(image), loc =0.0, scale=0.2)
        image_noised = np.add(image, noise)

        return image_noised

    def aug(image, idx):
        augmentation_dic = {0: enlarge(image, 1.2),
                            1: rotation(image),
                            2: random_bright_contrast(image),
                            3: gaussian_noise(image)}

        image = augmentation_dic[idx]
        return image


    p =[random.random() for m in range(4)] # 4 is number of augmentation operation
    for n in range(len(p)):
        if p[n] > 0.5:
            image = aug(image, n)

    return image



