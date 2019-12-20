### SET UP
import numpy as np
from scipy import signal, sparse
import skimage as sk
import skimage.filters as sf
import skimage.io as skio
from skimage.io import imsave, imshow, show
from skimage.color import grey2rgb

from operations import *
from utils import *

def main():

    impathM = 'sample_imgs/spline/mask2.jpg'
    imM = skio.imread(impathM)
    imM = sk.img_as_float(imM)

    impathToyMaskMix = 'sample_imgs/samples/toy_problem mask_mix.png'
    imToyMaskMix = skio.imread(impathToyMaskMix)
    imToyMaskMix = sk.img_as_float(imToyMaskMix)
    # viewImage(imToyMaskMix)

    impathToyCol = 'sample_imgs/samples/toy_problem_color.png'
    imToyCol = skio.imread(impathToyCol)
    imToyCol = sk.img_as_float(imToyCol)
    # viewImage(imToyMask)

    impathPengChick = 'sample_imgs/samples/penguin-chick.jpeg'
    imPengChick = skio.imread(impathPengChick)
    imPengChick = sk.img_as_float(imPengChick)
    # viewImage(imPengChick)

    impathPeng = 'sample_imgs/samples/penguin.jpg'
    imPeng = skio.imread(impathPeng)
    imPeng = sk.img_as_float(imPeng)
    # viewImage(imPeng)

    impathPengMask = 'sample_imgs/samples/penguin_mask.png'
    imPengMask = skio.imread(impathPengMask)
    imPengMask = sk.img_as_float(imPengMask)
    # viewImage(imPengMask)

    impathIm3 = 'sample_imgs/samples/im3.jpg'
    imIm3 = skio.imread(impathIm3)
    imIm3 = sk.img_as_float(imIm3)
    # viewImage(imIm3)

    toyProb = False

    targetYX = (1576, 1088)
    imTarg = imIm3
    imSrc = imPeng
    srcMask = imPengMask

    # targetYX = (0, 0)
    # imTarg = imToyCol
    # imSrc = imToyCol
    # srcMask = imToyMaskMix

    # targetYX = (0, 0)
    # imTarg = imT
    # imSrc = imT
    # srcMask = imToyMask
    # toyProb = True

    result = gradFusionFn(targetYX, imTarg, imSrc, srcMask, toyProb)

    testImage("toyProb____test.png", result, disp=False)

###

def getPixWCC(im, y, x, colorChannel = None):
    if colorChannel == None:
        return im[y, x]
    elif colorChannel < 3:
        return im[y, x, colorChannel]
    else:
        raise Exception("Unrecognized colorChannelCt:" + str(colorChannel))

def gradFusionFn(targetYX, imTarg, imSrc, srcMask, toyProb=False):
    if srcMask.ndim == 3:
        srcMask = np.squeeze(np.dsplit(srcMask, [1])[0], axis=2)

    newImSrc, newSrcMask = shiftObjectNMask(targetYX, imSrc, srcMask, imTarg)

    im2Var, var2Im, totalUnknowns, dictRow2Cols, dictCol2Rows, getColBorders, getRowBorders = makeIm2Var(newSrcMask)

    if imTarg.ndim == 3:
        maxHeightInd, maxWidthInd, colorChannelCt = imTarg.shape
    elif imTarg.ndim == 2:
        maxHeightInd, maxWidthInd = imTarg.shape
        colorChannelCt = 1
    else:
        raise Exception("...unrecognized ndim?")
    maxHeightInd -= 1
    maxWidthInd -= 1

    rHeight, rWidth = len(dictRow2Cols), len(dictCol2Rows)
    finalIm22 = np.matrix.copy(imTarg)

    # print(maxHeightInd, maxWidthInd)

    totalAFinished = False

    for colorChannel in range(colorChannelCt):
        if colorChannelCt == 1:
            colorChannel = None

        neg1_changeListY = []
        neg1_changeListX = []
        pos1_changeListY = []
        pos1_changeListX = []

        # def neg1(y, x):

        bList = []

        newLineCt = -1
        for y, cols in dictRow2Cols.items():
            # cols = dictRow2Cols[y]
            if len(cols) <= 1:
                continue

            for col in cols:
                if col == maxWidthInd or col in getRowBorders(y, "rtl"):
                    continue
                newLineCt += 1
                bList.append(getPixWCC(newImSrc, y, col + 1, colorChannel) - getPixWCC(newImSrc, y, col, colorChannel))
                if totalAFinished:
                    continue
                neg1_changeListY.append(newLineCt)
                neg1_changeListX.append(im2Var(y, col))

                pos1_changeListY.append(newLineCt)
                pos1_changeListX.append(im2Var(y, col + 1))

        for x, rows in dictCol2Rows.items():
            if len(rows) <= 1:
                continue

            for row in rows:  # in range(len(rows) - 1):
                if row == maxHeightInd or row in getColBorders(x, "rtl"):
                    continue
                newLineCt += 1
                bList.append(getPixWCC(newImSrc, row + 1, x, colorChannel) - getPixWCC(newImSrc, row, x, colorChannel))
                if totalAFinished:
                    continue
                neg1_changeListY.append(newLineCt)
                neg1_changeListX.append(im2Var(row, x))

                pos1_changeListY.append(newLineCt)
                pos1_changeListX.append(im2Var(row + 1, x))

        # print(neg1_changeListY)
        # print(neg1_changeListX)
        # print(pos1_changeListY)
        # print(pos1_changeListX)

        ## TOY PROB corner match
        if toyProb:
            newLineCt += 1
            pos1_changeListY.append(newLineCt)
            pos1_changeListX.append(im2Var(0, 0))
            bList.append(newImSrc[0, 0])

        #     ## original A construction
        #     newA = np.zeros((newLineCt + 1, totalUnknowns))

        #     newA[neg1_changeListY,neg1_changeListX] = -1
        #     newA[pos1_changeListY,pos1_changeListX] = 1

        b = np.array(bList)

        outReg_neg1_changeListY = []
        outReg_neg1_changeListX = []
        outReg_pos1_changeListY = []
        outReg_pos1_changeListX = []

        outReg_bList = []

        outReg_LineCt = newLineCt

        for row in dictRow2Cols:
            for i in getRowBorders(row, "ltr"):
                # print("before:", outReg_LineCt)
                outReg_LineCt += 1
                # outReg_bList.append(imTarg[row, i+1, colorChannel])
                outReg_bList.append(
                    getPixWCC(imTarg, row, i + 1, colorChannel) + getPixWCC(newImSrc, row, i, colorChannel) - getPixWCC(
                        newImSrc, row, i + 1, colorChannel))

                if totalAFinished:
                    continue
                outReg_pos1_changeListY.append(outReg_LineCt)
                outReg_pos1_changeListX.append(im2Var(row, i))
                # print("LTR row: ", outReg_LineCt, im2Var(row, i))
            for j in getRowBorders(row, "rtl"):
                outReg_LineCt += 1
                # outReg_bList.append(imTarg[row, j-1, colorChannel])
                outReg_bList.append(
                    getPixWCC(imTarg, row, j - 1, colorChannel) + getPixWCC(newImSrc, row, j, colorChannel) - getPixWCC(
                        newImSrc, row, j - 1, colorChannel))

                if totalAFinished:
                    continue
                outReg_pos1_changeListY.append(outReg_LineCt)
                outReg_pos1_changeListX.append(im2Var(row, j))
                # print("RTL row: ", outReg_LineCt, im2Var(row, j))
        for col in dictCol2Rows:
            for m in getColBorders(col, "ltr"):
                outReg_LineCt += 1
                # outReg_bList.append(imTarg[m+1, col, colorChannel])
                outReg_bList.append(
                    getPixWCC(imTarg, m + 1, col, colorChannel) + getPixWCC(newImSrc, m, col, colorChannel) - getPixWCC(
                        newImSrc, m + 1, col, colorChannel))

                if totalAFinished:
                    continue
                outReg_pos1_changeListY.append(outReg_LineCt)
                outReg_pos1_changeListX.append(im2Var(m, col))
            for n in getColBorders(col, "rtl"):
                outReg_LineCt += 1
                # outReg_bList.append(imTarg[n-1, col, colorChannel])
                outReg_bList.append(
                    getPixWCC(imTarg, n - 1, col, colorChannel) + getPixWCC(newImSrc, n, col, colorChannel) - getPixWCC(
                        newImSrc, n - 1, col, colorChannel))

                if totalAFinished:
                    continue
                outReg_pos1_changeListY.append(outReg_LineCt)
                outReg_pos1_changeListX.append(im2Var(n, col))

        #     # pre sparse A
        #     outReg_A = np.zeros((outReg_LineCt + 1, totalUnknowns))

        #     # print(outReg_pos1_changeListY)
        #     # print(outReg_pos1_changeListX)

        #     outReg_A[outReg_neg1_changeListY,outReg_neg1_changeListX] = -1
        #     outReg_A[outReg_pos1_changeListY,outReg_pos1_changeListX] = 1

        # newA
        outReg_b = np.array(outReg_bList)

        # print(b.shape, outReg_b.shape)

        ### FINAL FORM
        if not totalAFinished:
            # totalA = np.vstack((newA, outReg_A))
            final_neg1_changeListX = neg1_changeListX + outReg_neg1_changeListX
            final_neg1_changeListY = neg1_changeListY + outReg_neg1_changeListY
            final_pos1_changeListX = pos1_changeListX + outReg_pos1_changeListX
            final_pos1_changeListY = pos1_changeListY + outReg_pos1_changeListY

            assert len(final_pos1_changeListX) == len(final_pos1_changeListY)
            assert len(final_neg1_changeListX) == len(final_neg1_changeListY)

            final_all_changeListY = final_neg1_changeListY + final_pos1_changeListY
            final_all_changeListX = final_neg1_changeListX + final_pos1_changeListX
            final_all_dataList = [-1] * len(final_neg1_changeListY) + [1] * len(final_pos1_changeListY)

            # print(outReg_LineCt, totalUnknowns, ":", max(final_all_changeListY), max(final_all_changeListX))
            totalA = sparse.csr_matrix((final_all_dataList, (final_all_changeListY, final_all_changeListX)),
                                       (outReg_LineCt + 1, totalUnknowns))

            totalAFinished = True
        totalB = np.hstack((b, outReg_b))

        print("Starting LSQR, color channel:", colorChannel)
        resolve2222 = sparse.linalg.lsqr(totalA, totalB)
        print("Finished LSQR, color channel:", colorChannel)

        # finalIm22[0, 0] = 0
        # finalIm22[0, 1] = 0

        ## you have to figure out how to do this w color pics
        sol2222 = resolve2222[0]
        sol2222 = np.clip(sol2222, -1, 1)
        for varId, value in enumerate(sol2222):
            hei, wid = var2Im(varId)
            if colorChannelCt == 1:
                finalIm22[hei, wid] = value
            else:
                finalIm22[hei, wid, colorChannel] = value

        if colorChannelCt == 1:
            break

    return finalIm22


### MASK UTILS

def createNonZeroPairs(mask):
    # TODO extend
    if mask.ndim == 3:
        mask = np.squeeze(np.dsplit(mask, [1])[0], axis=2)
        # change 3 channel to 1 value per pixel,
        # specifically using the 0th channel's
    maskNonZero = mask.nonzero()

    #    return maskNonZero

    nonZeroIndices = []
    #    mask
    for i in range(len(maskNonZero[0])):
        nonZeroIndices.append((maskNonZero[0][i], maskNonZero[1][i]))

    return nonZeroIndices

def makeIm2Var(mask):
    nonZeroIndices = createNonZeroPairs(mask)

    maskHeight, maskWidth = mask.shape
    # print("weepweep: ", maskHeight, maskWidth)

    dictIm2Var = dict()
    dictVar2Im = dict()

    dictRow2Cols = dict()
    dictCol2Rows = dict()

    fTup2Key = (lambda tup: "%i-%i" % tup)
    fYX2Key = (lambda y, x: fTup2Key((y, x)))

    for i in range(len(nonZeroIndices)):
        pair = nonZeroIndices[i]
        row, col = pair

        if row not in dictRow2Cols:
            dictRow2Cols[row] = []
        dictRow2Cols[row].append(col)

        if col not in dictCol2Rows:
            dictCol2Rows[col] = []
        dictCol2Rows[col].append(row)

        key = fTup2Key(pair)
        dictIm2Var[key] = i
        dictVar2Im[i] = pair

    im2Var = (lambda y, x: dictIm2Var[fYX2Key(y, x)])
    var2Im = (lambda x: dictVar2Im[x])
    totalUnknowns = len(nonZeroIndices)

    dictCol2Borders = dict()
    dictRow2Borders = dict()

    def getColBorders(col, mode):
        if mode == "ltr":
            mode = 0
        elif mode == "rtl":
            mode = 1
        elif mode == "both":
            mode = 2
        else:
            raise Exception("unrecognized mode")
        return dictCol2Borders[col][mode]

    def getRowBorders(row, mode):
        if mode == "ltr":
            mode = 0
        elif mode == "rtl":
            mode = 1
        elif mode == "both":
            mode = 2
        else:
            raise Exception("unrecognized mode")
        return dictRow2Borders[row][mode]

    for col, rows in dictCol2Rows.items():
        dictCol2Borders[col] = findBorders(rows, maskHeight - 1)
    for row, cols in dictRow2Cols.items():
        dictRow2Borders[row] = findBorders(cols, maskWidth - 1)

    # print(dictIm2Var)
    return im2Var, var2Im, totalUnknowns, dictRow2Cols, dictCol2Rows, getColBorders, getRowBorders


def findBorders(row, maxIndex):
    #     if mode == "rtl":
    #         RTL = True
    #         LTR = False
    #     elif mode == "ltr":
    #         RTL = False
    #         LTR = True
    #     elif mode == "both":
    #         RTL = True
    #         LTR = True
    """
    input = [1, 2, 5, 6]
    output = [2, 5]
    # q: do we care about image boundary boundaries
    # a: no, it'll eff up the algo -- if you need image boundary stuff
    # add it separately on the algo
    """
    assert row[-1] <= maxIndex

    ltrBorders = set()
    rtlBorders = set()

    if len(row) == 0:
        return row

    minIndex = 0

    diff = [row[0] - (minIndex - 1)]
    for i in range(1, len(row)):
        diff.append(row[i] - row[i - 1])
    diff.append(maxIndex + 1 - row[-1])
    # print(diff)

    for ind, val in enumerate(diff):
        if val > 1:
            # if RTL:
            if ind - 1 >= 0:  # comment this out to get left to right edge
                rtlBorders.add(row[ind - 1])
            # if LTR:
            if ind < len(row):  # comment this out to get right to left edge
                ltrBorders.add(row[ind])

    return ltrBorders, rtlBorders, ltrBorders.union(rtlBorders)


def shiftObjectNMask(targetYX, imSrc, srcMask, imTarg):
    if imTarg.ndim == 3:
        targHeight, targWid, _ = imTarg.shape
    elif imTarg.ndim == 2:
        targHeight, targWid = imTarg.shape

    if imSrc.ndim == 3:
        srcHeight, srcWid, _ = imSrc.shape
    elif imSrc.ndim == 2:
        srcHeight, srcWid = imSrc.shape

    # print(srcMask.shape)
    # print(srcMask.shape)
    assert (srcHeight, srcWid) == srcMask.shape, str((srcHeight, srcWid)) + " != " + str(srcMask.shape)
    assert targWid and targHeight and srcWid and srcHeight

    rowStart = targetYX[0]
    rowEnd = targetYX[0] + srcHeight
    colStart = targetYX[1]
    colEnd = targetYX[1] + srcWid

    if rowStart < 0 or rowEnd > targHeight or colStart < 0 or colEnd > targWid:
        raise Exception("source img doesn't fit target img in the given location: %s" % str(imSrc))

    # print(targHeight, targWid)

    if imSrc.ndim == 3:
        newImSrc = np.zeros((targHeight, targWid, 3))
        newImSrc[rowStart:rowEnd, colStart:colEnd, 0] = imSrc[:, :, 0]
        newImSrc[rowStart:rowEnd, colStart:colEnd, 1] = imSrc[:, :, 1]
        newImSrc[rowStart:rowEnd, colStart:colEnd, 2] = imSrc[:, :, 2]
    elif imSrc.ndim == 2:
        newImSrc = np.zeros((targHeight, targWid))
        newImSrc[rowStart:rowEnd, colStart:colEnd] = imSrc

    newSrcMask = np.zeros((targHeight, targWid))
    newSrcMask[rowStart:rowEnd, colStart:colEnd] = srcMask

    return newImSrc, newSrcMask


# findBorders([0, 2, 4, 5, 6, 7], 8)
if __name__ == "__main__":
    main()