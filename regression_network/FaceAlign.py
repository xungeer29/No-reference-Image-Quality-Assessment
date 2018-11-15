# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

#=====================================FACE ALIGNED=======================================================================
RefShape = [-27.606326, -30.763604, 27.604702, -30.764311, 0.002382, 1.479138, -23.444776, 30.02466, 23.444018, 30.024117]

RefShape21 = [-27.606326, -30.763604,
              27.604702, -30.764311,
              0.002382, 1.479138,
              -23.444776, 30.02466,
              23.444018, 30.024117]

_RefShape21 = []

for i in range(21):
    _RefShape21.append(0.0)
    _RefShape21.append(0.0)

_RefShape21[7*2] = RefShape[0] + 64.0
_RefShape21[7*2+1] = RefShape[1] + 64.0
_RefShape21[10*2] = RefShape[2] + 64.0
_RefShape21[10*2+1] = RefShape[3] + 64.0
_RefShape21[12*2] = RefShape[4] + 64.0
_RefShape21[12*2+1] = RefShape[5] + 64.0
_RefShape21[16*2] = RefShape[6] + 64.0
_RefShape21[16*2+1] = RefShape[7] + 64.0
_RefShape21[18*2] = RefShape[8] + 64.0
_RefShape21[18*2+1] = RefShape[9] + 64.0

# print(_RefShape21)

def calcProcrustes(predict_shape, ref_shape ):
    rigid_transform = [0,0,0,0,0,0]
    X1 = 0
    Y1 = 0
    Z = 0
    C1 = 0
    C2 = 0

    for i in range(5):
        x1 = predict_shape[i*2]
        y1 = predict_shape[i*2+1]
        x2 = ref_shape[i*2]
        y2 = ref_shape[i*2+1]
        X1 += x1
        Y1 += y1
        Z += x2*x2 + y2*y2
        C1 += x1 * x2 + y1 * y2
        C2 += y1 * x2 - x1 * y2

    temp_a = C1 / Z
    temp_b = C2 / Z
    rigid_transform[0] = temp_a
    rigid_transform[1] = -temp_b
    rigid_transform[2] = temp_b
    rigid_transform[3] = temp_a
    rigid_transform[4] = X1 / 5
    rigid_transform[5] = Y1 / 5

    return rigid_transform

def cropAlignedFaceImage(rigidTransform, initImg, width = 128, bilinear = True):
    RefShapeX = 58 + 6 + (width - 128)/2
    RefShapeY = 58.3698 + 6 + (width - 128)/2

    imgW = initImg.shape[1]
    imgH = initImg.shape[0]

    r00 = rigidTransform[0]
    r01 = rigidTransform[1]
    r10 = rigidTransform[2]
    r11 = rigidTransform[3]
    tx = rigidTransform[4]
    ty = rigidTransform[5]

    y, x = np.mgrid[0:width, 0:width]

    initX = r00*(x-RefShapeX) + r01*(y-RefShapeY) + tx
    initY = r10*(x-RefShapeX) + r11*(y-RefShapeY) + ty
    initX = np.clip(initX, 0, imgW-2)
    initY = np.clip(initY, 0, imgH-2)

    if bilinear:
        cropImg = bilinear_interpolate(initImg, initX, initY)
    else:
        cropImg = nearest_interpolate(initImg, initX, initY)


    return cropImg

def bilinear_interpolate(im, x, y):
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def nearest_interpolate(im, x, y):
    x0 = np.floor(x+0.5).astype(int)
    y0 = np.floor(y+0.5).astype(int)
    Ia = im[ y0, x0 ]
    return Ia

def AlignedFace_from_5points(points, gray, width = 128, bilinear = True):
    rigid_transform = calcProcrustes(points, RefShape)
    crop_img = cropAlignedFaceImage(rigid_transform, gray, width, bilinear = bilinear)

    return np.uint8(crop_img)


def AlignedFace_rgb_from_5points(points, rgb, width = 128):
    rigid_transform = calcProcrustes(points, RefShape)

    crop_img_r = cropAlignedFaceImage(rigid_transform, rgb[:, :, 0], width)
    crop_img_g = cropAlignedFaceImage(rigid_transform, rgb[:, :, 1], width)
    crop_img_b = cropAlignedFaceImage(rigid_transform, rgb[:, :, 2], width)

    img_list = [crop_img_r, crop_img_g, crop_img_b]
    crop_rgb = np.asarray(img_list)
    crop_rgb = np.transpose(crop_rgb, [1, 2, 0])

    return np.uint8(crop_rgb)

def CropFace_from_bbox(bbox, gray, width = 128, bilinear = True):
    # rigid_transform = calcProcrustes(points, RefShape)
    # crop_img = cropAlignedFaceImage(rigid_transform, gray, width, bilinear = bilinear)
    pad = (bbox[3] - bbox[1]) * (width/128 - 1.0) / 2.0

    py0 = int(bbox[1] - pad)
    py1 = int(bbox[3] + pad)
    px0 = int(bbox[0] - pad)
    px1 = int(bbox[2] + pad)

    crop = gray[ max(py0,0): min(py1, gray.shape[0]-1), max(px0,0): min(px1, gray.shape[1]-1)]
    random = cv2.resize(crop,(width,width))


    return np.uint8(random)



if __name__ == '__main__':
    path = r'./assad_ahmadi_0001.jpg'
    img = cv2.imread(path, 0)
    pts = [106.669716, 111.415695, 141.406433, 108.801643, 134.098282, 135.460464, 114.398483, 160.126678, 142.901520, 156.720047]
    crop = AlignedFace_from_5points(pts, img, width=128, bilinear=True)
    crop = cv2.resize(crop,(128,128))
    cv2.imshow('tmp', np.uint8(crop))
    cv2.waitKey(0)