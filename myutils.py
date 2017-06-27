from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import matplotlib.image as mpimg
# import cv2
import os
from PIL import Image
import os
import matplotlib.image as mpimg



list_images = os.listdir("data/training/images")
list_segmentations = os.listdir("data/training/1st_manual")
import pickle
import scipy.misc


def show_test_image():
    filename = 'data/test/images/01_test.tif'
    img = mpimg.imread(filename)
    imgplot = plt.imshow(img)

# === Step 1: Initialize the Network

# === Step 2: Train the Network

def show_training_images():
    plt.figure(figsize=(20, 12))
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        filename = 'data/training/images/' + list_images[i - 1]
        img = mpimg.imread(filename)
        imgplot = plt.imshow(img)

def show_training_segmentations():
    plt.figure(figsize=(20, 12))
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        filename = 'data/training/1st_manual/' + list_segmentations[i - 1]
        img = mpimg.imread(filename)
        imgplot = plt.imshow(img, cmap=plt.get_cmap('gray'))

# Convert to grayscale

def convert_training_to_grayscale():
    if not os.path.exists('grayscale'):
        os.makedirs('grayscale')
        os.makedirs('grayscale/training')
        os.makedirs('grayscale/test')

    plt.figure(figsize=(20, 12))
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        filename = 'data/training/images/' + list_images[i - 1]
        img = mpimg.imread(filename)
        gs = 0.21 * img[:, :, 0] + 0.71 * img[:, :, 1] + 0.07 * img[:, :, 2]
        imgplot = plt.imshow(gs, cmap='gray')
        filename = 'grayscale/training/' + list_images[i - 1]
        scipy.misc.imsave(filename, gs)

def build_training_set(patchsize, n_patches_per_image):
    n_chosable_pixels = 296390
    X_train = np.empty([n_patches_per_image * 20, patchsize, patchsize])
    y_train = np.zeros(n_patches_per_image * 20)
    filenameGT = 'data/training/1st_manual/21_manual1.gif'
    gt = mpimg.imread(filenameGT)
    ind = 0
    for i in range(21, 41):
        filenameImg = 'grayscale/training/' + str(i) + '_training.tif'
        filenameGT = 'data/training/1st_manual/' + str(i) + '_manual1.gif'
        gs = mpimg.imread(filenameImg)  # grayscale image
        gt = mpimg.imread(filenameGT)  # groundtruth segmentation

        patchSemiSize = (patchsize - 1) / 2

        one_dim_indices = np.random.choice(n_chosable_pixels, n_patches_per_image)
        for j in range(0, n_patches_per_image):
            lineIndCenter = (one_dim_indices[j] // 535) + patchSemiSize
            colIndCenter = (one_dim_indices[j] % 535) + patchSemiSize
            patch = gs[int(lineIndCenter - patchSemiSize):int(lineIndCenter + patchSemiSize + 1),
                    int(colIndCenter - patchSemiSize):int(colIndCenter + patchSemiSize + 1)]
            X_train[ind] = patch/255

            if gt[int(lineIndCenter), int(colIndCenter)] == 255:
                y_train[ind] = 1

            ind = ind + 1

    X_train_new = np.empty([n_patches_per_image * 20, patchsize, patchsize])
    y_train_new = np.empty(n_patches_per_image * 20)
    arr = np.arange(n_patches_per_image * 20)
    np.random.shuffle(arr)
    X_train_new[arr] = X_train
    y_train_new[arr] = y_train

    plt.figure(figsize=(25, 16))
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        plt.imshow(X_train[i - 1], cmap=plt.get_cmap('gray'))
        plt.title('label=' + str(y_train[i - 1]))
    plt.show()

    return {'patches': X_train_new, 'labels':y_train_new}

    #filename = 'X_train' + str(patchsize) + '.p'
    #pickle.dump(X_train, open(filename, "wb"))
    #filename = 'y_train' + str(patchsize) + '.p'
    #pickle.dump(y_train, open(filename, "wb"))


def plot_history(history):
    plt.figure(figsize=(6,6))
    axes = plt.gca()
    vl = history.history['val_loss']
    ind = vl.index(min(vl))
    va = history.history['val_acc']
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(ind,vl[ind],color="g", marker="o")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

# === Step 3: Test the Network

def convert_test_to_grayscale():
    if not os.path.exists('grayscale'):
        os.makedirs('grayscale')
        os.makedirs('grayscale/training')
        os.makedirs('grayscale/test')

    filename = 'data/test/images/01_test.tif'
    img = mpimg.imread(filename)
    gs = 0.21 * img[:, :, 0] + 0.71 * img[:, :, 1] + 0.07 * img[:, :, 2]
    imgplot = plt.imshow(img)
    imgplot = plt.imshow(gs, cmap='gray')
    filename = 'grayscale/test/01_test.tif'
    scipy.misc.imsave(filename, gs)

def build_test_set(patchsize):
    filenameImg = "grayscale/test/01_test.tif"
    #img = mpimg.imread(filenameImg)
    #img = np.asarray(img)
    #gs = 0.21 * img[:, :, 0] + 0.71 * img[:, :, 1] + 0.07 * img[:, :, 2]
    gs = mpimg.imread(filenameImg)
    gs = gs/255
    nlines = gs.shape[0]
    ncols = gs.shape[1]
    npatches = (nlines - patchsize + 1) * (ncols - patchsize + 1)
    X_test = np.empty([npatches, patchsize, patchsize])

    beg = (patchsize - 1) / 2 + 1  # index when starting from 1
    endl = nlines - (patchsize - 1) / 2  # index when starting from 1
    endc = ncols - (patchsize - 1) / 2  # index when starting from 1

    beg = int(beg)
    endl = int(endl)
    endc = int(endc)
    ind = 0
    for l in range(beg, endl + 1):
        for c in range(beg, endc + 1):
            l1 = l - (patchsize - 1) / 2 - 1  # index when starting from 0
            l2 = l + (patchsize - 1) / 2 - 1 + 1  # index when starting from 0
            c1 = c - (patchsize - 1) / 2 - 1  # index when starting from 0
            c2 = c + (patchsize - 1) / 2 - 1 + 1  # index when starting from 0
            patch = gs[int(l1):int(l2), int(c1):int(c2)]
            X_test[ind] = patch
            ind = ind + 1

    plt.figure(figsize=(20, 12))
    for i in range(1, 20):
        plt.subplot(4, 5, i)
        plt.imshow(X_test[i*10 - 1], cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

    return {'patches': X_test}

def show_test_segmentation():
    filename = 'data/test/1st_manual/01_manual1.gif'
    img = mpimg.imread(filename)
    plt.figure(figsize=(8, 6))
    plt.title("Manual expert segmentation")
    imgplot = plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

def quantitative_evaluation(binarymask, heatmap):
    from sklearn import metrics
    import pandas as pd

    filenameGT = 'data/test/1st_manual/01_manual1.gif'
    gt = mpimg.imread(filenameGT)
    binarymask.shape

    pad = np.zeros((15, 535))
    padded_heatmap = np.concatenate((pad, heatmap, pad), axis=0)
    padded_binarymask = np.concatenate((pad, binarymask, pad), axis=0)
    pad = np.zeros((584, 15))
    padded_heatmap = np.concatenate((pad, padded_heatmap, pad), axis=1)
    padded_binarymask = np.concatenate((pad, padded_binarymask, pad), axis=1)


    filepath = 'data/test/mask/01_test_mask.gif'
    fov = mpimg.imread(filepath)

    n_pixels_fov = np.sum(fov) / 255
    fov_heatmap = np.zeros((int(n_pixels_fov), 1))
    fov_binarymask = np.zeros((int(n_pixels_fov), 1))
    fov_gt = np.zeros((int(n_pixels_fov), 1))
    ind = 0
    for i in range(1, 584):
        for j in range(1, 565):
            if fov[i][j] == 255:
                fov_heatmap[ind] = padded_heatmap[i][j]
                fov_binarymask[ind] = padded_binarymask[i][j]
                fov_gt[ind] = gt[i][j]
                ind = ind + 1

    y = fov_gt / 255
    scores = fov_heatmap
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    auc = round(auc, 4)

    scores = fov_binarymask
    acc = metrics.accuracy_score(y, scores)
    acc = round(acc, 4)
    tp = np.sum(np.multiply(y, scores))
    p = np.sum(y)
    tpr = tp / p
    sens = tpr
    sens = round(sens, 4)
    y_perm = 1 - y
    scores_perm = 1 - scores
    tn = np.sum(np.multiply(y_perm, scores_perm))
    n = np.sum(y_perm)
    tnr = tn / n
    tnr
    spec = tnr
    spec = round(spec, 4)
    
    pan = 1 - np.sum(y) / len(y)  # accuracy (=specificity) when predicting all negative
    pan = round(pan, 4)

    d = {'AUC': ['Undefined', auc],
         'Accuracy': [pan, acc],
         'Sensibility': ['Undefined', sens],
         'Specificity': [pan, spec]}

    dataframe = pd.DataFrame(d, index=['Predict all negative', 'Proposed method'])

    return dataframe
