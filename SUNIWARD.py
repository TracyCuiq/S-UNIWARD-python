import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve2d
import math
from scipy import misc
import os
from PIL import Image
from numba import jit
import cv2
import scipy.misc

np.set_printoptions(threshold=np.inf)

def S_UNIWARD(coverPath, payload):
    sgm = 1
    ## Get 2D wavelet filters - Daubechies 8
    # 1D high pass decomposition filter

    hpdf_list = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053,
                 -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539,
                 - 0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768]

    # 1D low pass decomposition filter
    hpdf_len = range(0, len(hpdf_list))
    hpdf_list_reverse = hpdf_list[::-1]
    lpdf_list = hpdf_list
    for i in range(len(hpdf_list)):
        lpdf_list[i] = ((-1) ** hpdf_len[i]) * hpdf_list_reverse[i]
    hpdf_array = np.array([hpdf_list])
    lpdf_array = np.array([lpdf_list])
    lpdf = lpdf_array.reshape(len(lpdf_list), 1)
    hpdf = hpdf_array.reshape(len(hpdf_list), 1)
    # construction of 2D wavelet filters
    F1 = lpdf * hpdf_array
    F2 = hpdf * lpdf_array
    F3 = hpdf * hpdf_array

    W_F = np.zeros((F1.shape[0], F1.shape[0], 3))
    W_F[:, :, 0] = F1
    W_F[:, :, 1] = F2
    W_F[:, :, 2] = F3

    ## Get embedding costs
    # initialization
    cover = scipy.misc.imread(coverPath, flatten=False, mode='RGB')
    wetCost = 100000000
    k, l, _ = cover.shape

    # add padding
    S1, _1 = F1.shape
    S2, _2 = F2.shape
    S3, _3 = F3.shape

    padSize = max(S1, S2, S3)
    coverPadded = np.zeros((k + padSize * 2, l + padSize * 2, 3))
    for i in range(3):
        coverPadded[:, :, i] = np.lib.pad(cover[:, :, i], padSize, 'symmetric')
    xi = np.zeros((k + padSize * 2, l + padSize * 2, 3))
    x = np.zeros((k, l, 3))
    for i in range(3):
        # compute residual
        R = convolve2d(coverPadded[:, :, i], W_F[:, :, i], mode='same')
        xi[:, :, i] = convolve2d(1. / (np.abs(R) + sgm), np.rot90(abs(W_F[:, :, i]), 2), mode='same')
        # correct the suitability shift if filter size is even
        if S1 % 2 == 0:
            xi[:, :, i] = np.roll(xi[:, :, i], [1, 0])
            xi[:, :, i] = np.roll(xi[:, :, i], [0, 1])
        # remove padding
        S_xi, __xi = xi[:, :, i].shape
        x[:, :, i] = xi[(S_xi - k) / 2: -(S_xi - k) / 2, (__xi - l) / 2: -(__xi - l) / 2, i]

    # compute embedding costs \rho
    rho = np.zeros((k, l))
    rho = x[:, :, 0] + x[:, :, 1] + x[:, :, 2]

    # adjust embedding costs
    a, b = np.where(rho > wetCost)
    for i in range(len(a)):
        rho[a[i], b[i]] = wetCost  # threshold on the costs
    a, b = np.where(np.isnan(rho))
    for i in range(len(a)):
        rho[a[i], b[i]] = wetCost  # if all xi{} are zero threshold the cost

    #k, k_ = rho.shape
    rhoP1 = np.zeros((k, l, 3))
    rhoM1 = np.zeros((k, l, 3))


    for i in range(3):
        rhoP1[:,:,i] = rho
        rhoM1[:,:,i] = rho
 
    #a, b, c = np.where(cover - 255.0 <= 0.1)
    a, b, c = np.where(cover == 255)

    for i in range(len(a)):
        rhoP1[a[i], b[i], c[i]] = wetCost  # do not embed +1 if the pixel has max value

    #a, b, c = np.where(cover - 0 <= 0.1)
    a, b, c = np.where(cover == 0)
    for i in range(len(a)):
        rhoM1[a[i], b[i], c[i]] = wetCost  # do not embed -1 if the pixel has min value

    
    ## Embedding simulator ##
    cover_len = len(cover[:, :, 0]) * len(cover[:, :, 0])
    stego = cover

    print(rhoP1)

    for i in range(3):
        stego[:, :, i] = EmbeddingSimulator_singel(cover[:, :, i], rhoP1[:, :, i], rhoM1[:, :, i], payload * cover_len,
                                                   fixEmbeddingChanges=False)
    return stego


# TODO
def EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges=False):
    cover_len = len(x[:, :, 0]) * len(x[:, :, 0])
    l = cal_lambda(rhoP1, rhoM1, m, cover_len)

    randChange = random.random(x.shape)
    y = x


def EmbeddingSimulator_singel(x, rhoP1, rhoM1, m, fixEmbeddingChanges=False):
    w, h = x.shape
    cover_len = (w * h)
    l = cal_lambda_(rhoP1, rhoM1, m, cover_len)
    shape = rhoP1.shape
    pChangeP1 = [(math.exp(-l * rhoP1[i][j])) / (1 + math.exp(-l * rhoP1[i][j]) + math.exp(-l * rhoM1[i][j]))
                 for j in range(shape[1]) for i in range(shape[0])]
    pChangeM1 = [(math.exp(-l * rhoM1[i][j])) / (1 + math.exp(-l * rhoP1[i][j]) + math.exp(-l * rhoM1[i][j]))
                 for j in range(shape[1]) for i in range(shape[0])]
    pChangeP1_array = np.array(pChangeP1).reshape(shape[1], shape[0]).T
    pChangeM1_array = np.array(pChangeM1).reshape(shape[1], shape[0]).T
    if fixEmbeddingChanges == True:
        np.random.seed(139187)
    randChange = np.random.rand(w, h)
    y = x

    arr0, _0 = np.where(randChange < pChangeP1_array)
    for i in range(len(arr0)):
        y[arr0[i]][_0[i]] += 1

    arr1, _1 = np.where((randChange >= pChangeP1_array) & (randChange < pChangeP1_array + pChangeM1_array))
    for i in range(len(arr1)):
        y[arr1[i]][_1[i]] -= 1
    return y


# TODO
def cal_lambda(rhoP1, rhoM1, message_length, n):
    l3 = 1e+3
    m3 = math.ceil(message_length)
    iterations = 0
    while m3 > message_length:
        pP1 = rhoP1
        pM1 = rhoM1
        shape = pP1.shape
        l3 = l3 * 2
        pP1 = [
            (math.exp(-l3 * rhoP1[i][j][k])) / (1 + math.exp(-l3 * rhoP1[i][j][k]) + math.exp(-l3 * rhoM1[i][j][k]))
            for k in range(shape[2]) for j in range(shape[1]) for i in range(shape[0])]  # list
        pM1 = [
            (math.exp(-l3 * rhoM1[i][j][k])) / (1 + math.exp(-l3 * rhoP1[i][j][k]) + math.exp(-l3 * rhoM1[i][j][k]))
            for k in range(shape[2]) for j in range(shape[1]) for i in range(shape[0])]  # list

        pP1_array = (np.array(pP1)).reshape(shape[0], shape[1], shape[2])
        pM1_array = (np.array(pM1)).reshape(shape[0], shape[1], shape[2])

        m3 = ternary_entropyf_4list(pP1, pM1)
        iterations = iterations + 1
        if iterations > 10:
            return l3

    return 0


def cal_lambda_(rhoP1, rhoM1, message_length, n):
    l3 = 1e+3
    m3 = math.ceil(message_length)
    iterations = 0
    while m3 > message_length:
        pP1 = rhoP1
        pM1 = rhoM1
        # shape = lambda x: pP1.shape if pP1.shape == pM1.shape else 0
        shape = pP1.shape
        l3 = l3 * 2
        pP1 = [(math.exp(-l3 * rhoP1[i][j])) / (1 + math.exp(-l3 * rhoP1[i][j]) + math.exp(-l3 * rhoM1[i][j]))
               for j in range(shape[1]) for i in range(shape[0])]  # list
        pM1 = [(math.exp(-l3 * rhoM1[i][j])) / (1 + math.exp(-l3 * rhoP1[i][j]) + math.exp(-l3 * rhoM1[i][j]))
               for j in range(shape[1]) for i in range(shape[0])]  # list

        pP1_array = (np.array(pP1)).reshape(shape[1], shape[0]).T
        pM1_array = (np.array(pM1)).reshape(shape[1], shape[0]).T

        m3 = ternary_entropyf_4list(pP1, pM1)
        iterations = iterations + 1
        if iterations > 10:
            return l3
        l1 = 0
        m1 = n
        l = 0
        alpha = message_length / n
        # limit search to 30 iterations
        # and require that relative payload embedded is roughly within 1/1000 of the required relative payload
        while (m1 - m3) / n > alpha / 1000.0 and iterations < 30:
            l = l1 + (l3 - l1) / 2
            pP1 = [(math.exp(-l * rhoP1[i][j])) / (1 + math.exp(-l * rhoP1[i][j]) + math.exp(-l * rhoM1[i][j]))
                   for j in range(shape[1]) for i in range(shape[0])]
            pM1 = [(math.exp(-l * rhoM1[i][j])) / (1 + math.exp(-l * rhoP1[i][j]) + math.exp(-l * rhoM1[i][j]))
                   for j in range(shape[1]) for i in range(shape[0])]
            m2 = ternary_entropyf_4list(pP1, pM1)
            if m2 < message_length:
                l3 = l
                m3 = m2
            else:
                l1 = l
                m1 = m2

            iterations = iterations + 1
    return 0


def ternary_entropyf(pP1_, pM1_):
    p0 = pP1_
    shape = p0.shape
    p0 = [1 - pP1_[i][j] - pM1_[i][j] for j in range(shape[1]) for i in range(shape[0])]
    ptemp = np.concatenate([[p0], [pP1_], [pM1_]])
    _, m, n = ptemp.shape
    p = np.reshape(ptemp, _ * m * n, 1)

    H = (-(p[i] * math.log(p[i])) for i in range(_ * m * n))
    Ht = sum(H)
    return Ht


def ternary_entropyf_4list(pP1_, pM1_):
    p0 = [1 - pP1_[i] - pM1_[i] for i in range(len(pP1_))]
    p = p0 + pP1_ + pM1_
    Ht = 0

    for i in range(len(p)):
        if p[i] != 0:
            H = -(p[i] * math.log(p[i]))
            Ht += H

    # Ht = sum(H)
    return Ht


coverPath = './sample'
stegoPath = './stego'

for home, dirs, files in os.walk(coverPath):
    for file in files:
        if not file.startswith('.'):
            imgpath = os.path.join(home, file)
            print(imgpath)
            #img = misc.imread(imgpath)
            img = Image.open(imgpath)
            #if img.ndim == 3:
            if len(img.split())== 3:
                stego = S_UNIWARD(imgpath, 0.4)
                stegoname = os.path.join(stegoPath, file)
                misc.imsave(stegoname, stego)
                #misc.imsave(stegoname, stego-img)
                plt.subplot(121)
                plt.imshow(img)
                plt.subplot(122)
                plt.imshow(stego)
                plt.show()

