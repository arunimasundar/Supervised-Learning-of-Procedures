import glob
import cv2
import pytesseract
import re
import numpy as np
import speechtotext as sp
import extract as ep
import comp as cp
import os

def academic(video):

    files = glob.glob('ocr frames/*')
    for f in files:
        os.remove(f)
    
    file = open("academic_video_output.txt","r+")
    file. truncate(0)
    file. close()

    file = open("ocr_notremoved.txt","r+")
    file. truncate(0)
    file. close()



    ep.extract(video)
    cp.comp()
    k=0
    check=True
    images = [cv2.imread(file) for file in glob.glob(r"ocr frames\*.jpg")]
    print(images)
    def gray(img):
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(r"./preprocess/img_gray.png",img)
        return img

    # blur
    def blur(img) :
        img_blur = cv2.GaussianBlur(img,(5,5),0)
        cv2.imwrite(r"./preprocess/img_blur.png",img)    
        return img_blur

    # threshold
    def threshold(img):
        #pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
        img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
        cv2.imwrite(r"./preprocess/img_threshold.png",img)
        return img
    # text detection
    def contours_text(orig, img, contours,k,check):
        for cnt in contours: 
            x, y, w, h = cv2.boundingRect(cnt) 

            # Drawing a rectangle on copied image 
            rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2) 
            
            # cv2.imshow('cnt',rect)
    #         cv2.waitKey()

            # Cropping the text block for giving input to OCR 
            cropped = orig[y:y + h, x:x + w] 

            # Apply OCR on the cropped image 
            config = ('-l eng --oem 1 --psm 3')
            text = pytesseract.image_to_string(cropped, config=config) 
            
            with open('ocr_notremoved.txt',mode ='a+') as file: 
                # print('yy', type(text))
                # if text==r"[\w\[\]`!@#$%\^&*()={}:;<>+'-]*":

                # if text!="/\n|\s{2,}/g":
                #     print('xx', text)
                #     fname=str("Step"+" "+str(k)+":")
                #     file.write(fname)
                #     file.write(text)




                # print('xx', text)
                if check:
                    file.write("\n")
                    file.write("\n")
                    fname=str("Step"+" "+str(k)+":")
                    file.write(fname) 
                    file.write("\n")
                    file.write("\n")

                file.write(text.strip()) 
                check=False


    #             file.write("\n") 
    #        print("ready!")

            # print(text)


    for img in images:
        # Finding contours
        print("in loop") 
        orig=img
        im_gray = gray(img)
        im_blur = blur(im_gray)
        im_thresh = threshold(im_blur)
        check=True

        contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        k=k+1
        contours_text(orig,im_thresh,contours,k,check)
        # cv2.destroyAllWindows()
            

    filename = 'ocr_notremoved.txt'
    file1 = open(filename, 'rt')

    lines = file1.readlines()
    with open('academic_video_output.txt',mode ='a+') as file2: 

        for line in lines:

            if not str(line).startswith(">>"):
                print(line)
    #             line.replace(">>","")
                file2.write(line) 
        
        sp.stt(video)
        fs=open("speechtotext.txt",'r+')
        speech=fs.read()

        file2.write("\n")
        file2.write("\n")
        file2.write("Additional Notes:")
        file2.write("\n")
        file2.write(speech)

    file1.close()


