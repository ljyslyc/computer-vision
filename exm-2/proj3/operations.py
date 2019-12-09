from scipy import signal
import gradFusion
from align_image_code import align_images, align_images_w_pts

from enum import Enum

## custom
from utils import *

##################
# OPERATIONS
##################

#### UNSHARP

def unsharpOp_3CC(im, alpha, sigma):
    result = np.zeros(im.shape)

    for i in range(3):
        result[:, :, i] = unsharpOp_1CC(im[:, :, i], alpha, sigma)

    return result

def unsharpOp_1CC(im, alpha, sigma):
    gaussKernel = makeGaussKernel(sigma)

    unitImpulseKernel = signal.unit_impulse(gaussKernel.shape, idx="mid")

    totalKernel = unitImpulseKernel * (1 + alpha) - (gaussKernel * alpha)

    result = signal.convolve2d(im[:, :], totalKernel, mode="same")

    result = np.clip(result, -1, 1)

    return result

def unsharpOp(im, alpha = 1.0, sigma = 10):
    if im.ndim == 3:
        return unsharpOp_3CC(im, alpha, sigma)
    elif im.ndim == 2:
        return unsharpOp_1CC(im, alpha, sigma)
    else:
        raise Exception("Unknown color channels set up")

#### GAUSS BLUR

def gaussBlurOp_1CC(im, sigma=10):
    gaussKernel = makeGaussKernel(sigma)

    result = signal.convolve2d(im, gaussKernel, mode="same", boundary="symm")

    # testImage("testGaussBlurOp.jpg", result)
    return result

def gaussBlurOp_3CC(im, sigma=10):
    assert im.ndim == 3

    result = []

    for i in range(3):
        result.append(gaussBlurOp_1CC(im[:, :, i], sigma))

    return np.dstack(result)

# returns a Gauss kernel -- desired MxM size MUST have M = an odd number
def makeGaussKernel(lowSig = 1, kernelSize = None):
    if kernelSize == None:
        kernelSize = int(np.ceil(lowSig * 3))
    assert kernelSize > 0
    # assert kernelSize % 2 != 0

    absEdgeVal = kernelSize // 2

    result = np.empty([kernelSize, kernelSize])

    sum = 0
    for preU in range(kernelSize):
        u = preU - absEdgeVal
        for preV in range(kernelSize):
            v = preV - absEdgeVal
            h = 1 / (2 * np.pi * (lowSig ** 2)) * np.exp(-1 * (u ** 2 + v ** 2) / (lowSig ** 2))
            result[preU, preV] = h
            sum += h

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            result[y, x] /= sum

    return result

#### HYBRID IMAGE

def hybridImageMidstep(im, sigma): ## doesn't work, so leave this out
    # image - gauss blurred image
    gaussKernel = makeGaussKernel(lowSig=sigma)
    unitImpulseKernel = signal.unit_impulse(gaussKernel.shape, idx="mid")

    totalKernel = unitImpulseKernel - gaussKernel
    # totalKernel = gaussKernel

    result = signal.convolve2d(im, totalKernel, mode="same")
    return result

def hybridImageFn(im1, im2, sigma1, sigma2):
    assert im1.ndim == im2.ndim

    if im1.ndim == 2:
        firstIm = im1[:, :]
        secondIm = im2[:, :]
        fin1 = gaussBlurOp_1CC(firstIm, sigma1)
        fin2 = secondIm - gaussBlurOp_1CC(secondIm, sigma2)
        res = np.dot(fin1 + fin2, 1 / 2)
        return res

    res = np.zeros(im1.shape)

    for i in range(3):
        firstIm = im1[:, :, i]
        secondIm = im2[:, :, i]
        fin1 = gaussBlurOp_1CC(firstIm, sigma1)
        fin2 = secondIm - gaussBlurOp_1CC(secondIm, sigma2)
        res[:, :, i] = np.dot(fin1 + fin2, 1/2)

    return res

def hybridImageOp(im1, im2, sigma1, sigma2, alignPts=None):
    if alignPts == None:
        im1_aligned, im2_aligned = align_images(im1, im2)
    else:
        im1_aligned, im2_aligned = align_images_w_pts(im1, im2, alignPts)

    hybrid = hybridImageFn(im1_aligned, im2_aligned, sigma1, sigma2)

    return hybrid



#### PYRAMID

class PyramidMode(Enum):
    Gaussian = "gauss"
    Laplacian = "laplacian"

def pyramidsOp(im, levels, sigma, mode = PyramidMode.Gaussian):
    if mode == PyramidMode.Gaussian:
        return gaussStackOp_3D(im, levels, sigma)
    elif mode == PyramidMode.Laplacian:
        return laplacianPyrOp_3D(im, levels, sigma)

def gaussStackOp_3D(im, levels, sigma):
    assert levels > 0
    #inclusive of original img, at layer indexed 0

    result = []
    # newLayer = (lambda: np.zeros(im.shape))

    for i in range(levels+1):
        if i == 0:
            result.append(im)
            continue
        # currLayer = newLayer()
        currLayer = gaussBlurOp_3CC(result[i - 1], sigma)
        result.append(currLayer)

    return np.array(result)

def laplacianPyrOp_3D(im, levels, sigma, scaleB = False):
    gaussStack = gaussStackOp_3D(im, levels, sigma)

    for i in range(levels):
        res = gaussStack[i] - gaussStack[i+1]
        if scaleB:
            finalCurrLayer = (res - res.min()) / (res.max() - res.min())
        else:
            finalCurrLayer = res
        gaussStack[i] = finalCurrLayer

    return gaussStack


#### MULTI RES

def scaler(LM):  # scales to 0 1
    return np.dot(LM - LM.min(), 1 / (LM.max() - LM.min()))  # * 2 - 1

def scalerNN(LM):  # scales to 0 1
    return np.dot(LM - LM.min(), 1 / (LM.max() - LM.min())) * 2 - 1

def multiResBlendOp(im1, im2, mask, levels, sigma):
    assert im1.shape == im2.shape == mask.shape

    L1 = laplacianPyrOp_3D(im1, levels, sigma)
    L2 = laplacianPyrOp_3D(im2, levels, sigma)
    LM = gaussStackOp_3D(mask, levels, sigma)
    LM1 = LM
    LM2 = (1 - LM1)

    L1_post = LM1 * L1
    L2_post = LM2 * L2

    finalL = L1_post + L2_post

    tes = np.zeros(L1[0].shape)

    for i in range(len(L1)):
        tes += finalL[i]

    tes2 = np.clip(tes, -1, 1)

    return tes2

#### GRAD FUSION -- this is a wrapper, because this was like, 7 functions

def gradFusionOp(targetYX, imTarg, imSrc, srcMask, toyProb=False):
    return gradFusion.gradFusionFn(targetYX, imTarg, imSrc, srcMask, toyProb)