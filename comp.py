from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title,k):

    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # if(s<0.95):
    #     path=r"C:\Users\Arunima\Desktop\OCR\ocr select"
    #     filename3=str("img"+str(k)+".jpg")
    #     cv2.imwrite(os.path.join(path , filename3),imageA)
        # k=k+1
    # setup the figure
    # fig = plt.figure(title)
    # plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(imageA, cmap = plt.cm.gray)
    # plt.axis("off")
    
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(imageB, cmap = plt.cm.gray)
    # plt.axis("off")
    # # show the images
    # plt.show()
    return s

def comp():
        
    folder=r"ocr frames"

    list1 = os.listdir(folder) # dir is your directory path
    number_files = len(list1)
    print(number_files-1)
    # print number_files
    for i in range(0,number_files-1):
        filename=str("img"+str(i)+".jpg")
        filename2=str("img"+str(i+1)+".jpg")
        original = cv2.imread(os.path.join(folder,filename))
        contrast = cv2.imread(os.path.join(folder,filename2))
        # cv2.imshow('rect',original)
        # cv2.imshow('rect2',contrast)


        # convert the images to grayscale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
        # shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

        # initialize the figure
        fig = plt.figure("Images")
        images = ("Original", original), ("Contrast", contrast)

        # # loop over the images
        # for (i, (name, image)) in enumerate(images):
        #     # show the image
        #     ax = fig.add_subplot(1, 3, i + 1)
        #     ax.set_title(name)
        #     plt.imshow(image, cmap = plt.cm.gray)
        #     plt.axis("off")

        # # show the figure
        # plt.show()

        # compare the images
        # compare_images(original, original, "Original vs. Original")
        s=compare_images(original, contrast, "Original vs. Contrast",i)
        # compare_images(original, shopped, "Original vs. Photoshopped")

        # print(s)

        if(s>0.95):
            os.remove(os.path.join(folder,filename))
            # path=r"C:\Users\Arunima\Desktop\OCR\ocr select"
            # filename3=str("img"+str(i)+".jpg")
            # cv2.imwrite(os.path.join(path , filename3),original)

