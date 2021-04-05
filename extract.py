import cv2
# Opens the Video file

def extract(video):
    
    cap= cv2.VideoCapture(video)
    i=0
    j=0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break
        if j%90==0:
            fn = 'ocr frames/img'+str(i)+'.jpg'
            cv2.imwrite(fn,frame)
            i+=1
        j+=1
    
    cap.release()
    cv2.destroyAllWindows()