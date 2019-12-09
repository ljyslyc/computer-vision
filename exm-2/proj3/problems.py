from utils import *
from operations import *
from skimage.color import rgb2gray

pathToImages = dict()
def saveImageBuffer(x, y):
    assert type(x) == str
    pathToImages[x] = y

def j(names):
    return "_".join([str(x) for x in names])

def printImageBuffer():
    for name, result in pathToImages.items():
        fname = "%s_%s" % (name, "result")
        printImage(fOutputDirectory, fname + fFormat, result, disp=False)

def outputPrint(name, im, disp=False):
    printImage(fOutputDirectory, name + fFormat, im, disp)

#########

def prob1d1():
    outputPrint("prob1d1_" + imN_bluelatte, unsharpOp(im_bluelatte))
    outputPrint("prob1d1_" + imN_girlface, unsharpOp(im_girlface))

    return

def prob1d2():

    alignPts1 = ((299.0064935064936, 348.3138528138527), (443.07575757575773, 333.0757575757574), (606.3571428571429, 289.21428571428555), (754.9285714285716, 369.21428571428555))
    result = hybridImageOp(im_DerekPicture, im_nutmeg, 20, 21, alignPts=alignPts1)

    outputPrint("prob1d2", result)

    # alignPts2 = ((377.04838709677415, 254.13225806451612), (500.7903225806451, 254.13225806451612),
    #  (220.95670995670991, 279.8463203463203), (358.6190476190476, 282.4437229437229))
    # repeater(im_woman1b, imN_woman1b, im_redpanda, imN_redpanda, 10, 50, alignPts2)
    # repeater(im_man, imN_man, im_cow, imN_cow, 10, 25)

    return

def prob1d3():
    result1 = gaussStackOp_3D(im_monalisa, 5, 10)
    for x in range(len(result1)):
        res1 = result1[x]
        outputPrint("prob1d3_gauss" + str(x), res1)

    result2 = laplacianPyrOp_3D(im_monalisa, 5, 10, scaleB=True)
    for x in range(len(result2)):
        res2 = result2[x]
        outputPrint("prob1d3_lap" + str(x), res2)

def prob1d4():
    # result = multiResBlendOp(im_soccer_s, im_volleyball_s, im_volley_mask_22, 5, 10)
    result = multiResBlendOp(im_apple, im_orange, im_mask2, 5, 10)

    outputPrint("prob1d4", result)
    return

def prob2d1():
    targetYX = (0, 0)
    imTarg = imT
    imSrc = imT
    srcMask = imToyMask
    toyProb = True

    result = gradFusionOp(targetYX, imTarg, imSrc, srcMask, toyProb)

    printImage(fOutputDirectory, "prob2d1" + fFormat, result, disp=False)

def prob2d2():
    targetYX = (333, 400)
    imTarg = im_desert
    imSrc = im_polar_bear
    srcMask = im_polar_bear_mask
    toyProb = False

    result = gradFusionOp(targetYX, imTarg, imSrc, srcMask, toyProb)

    printImage(fOutputDirectory, "prob2d2" + fFormat, result, disp=False)



## READING IMAGES

im_bluelatte, imN_bluelatte = readImageNName('sample_imgs/bluelatte.jpg')
im_girlface, imN_girlface = readImageNName('sample_imgs/girlface.bmp')
im_DerekPicture, imN_DerekPicture = readImageNName('sample_imgs/DerekPicture.jpg')
im_nutmeg, imN_nutmeg = readImageNName('sample_imgs/nutmeg.jpg')
im_redpanda, imN_redpanda = readImageNName('sample_imgs/redpanda.jpg')
im_woman1b, imN_woman1b = readImageNName('sample_imgs/woman1b.jpg')
im_man, imN_man = readImageNName('sample_imgs/man.jpg')
im_cow, imN_cow = readImageNName('sample_imgs/cow.jpg')
im_polar_bear, imN_polar_bear = readImageNName('sample_imgs/polar_bear.jpg')
im_polar_bear_mask, imN_polar_bear_mask = readImageNName('sample_imgs/polar_bear_mask.png')
im_desert, imN_desert = readImageNName('sample_imgs/desert.jpg')
im_monalisa, imN_monalisa = readImageNName('sample_imgs/monalisa.jpg')

im_soccer_s, imN_soccer_s = readImageNName('sample_imgs/soccer_s.jpg')
im_volleyball_s, imN_volleyball_s = readImageNName('sample_imgs/volleyball_s.jpg')
im_volley_mask_22, imN_volley_mask_22 = readImageNName('sample_imgs/volley_mask_22.png')

im_apple, imN_apple = readImageNName('sample_imgs/spline/apple.jpeg')
im_orange, imN_orange = readImageNName('sample_imgs/spline/orange.jpeg')
im_mask2, imN_mask2 = readImageNName('sample_imgs/spline/mask2.jpg')


impathToy = 'sample_imgs/toy_problem.png'
imT = skio.imread(impathToy)
imT = sk.img_as_float(imT)

impathToyMask = 'sample_imgs/toy_problem mask.png'
imToyMask = skio.imread(impathToyMask)
imToyMask = sk.img_as_float(imToyMask)
# viewImage(imToyMask)