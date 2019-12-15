import problems

# custom packages
import operations # apply, Operation
import utils

### INPUT

# input file1
im1, imname1 = utils.readImageNName('sample_imgs/DerekPicture.jpg')

# input file2
im2, imname2 = utils.readImageNName('sample_imgs/nutmeg.jpg')

### LOGIC, RUN THINGS HERE!

utils.printImage("main_test.png", operations.hybridImageOp(im1, im2, 20, 21))
