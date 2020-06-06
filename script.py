#!bin/python3

# OpenCV program to perform Edge detection in real time 
# import libraries of python OpenCV  
# where its functionality resides 
import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
import sys

src = cv2.imread('', cv2.IMREAD_UNCHANGED) 


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
  
# capture frames from a camera 
cap = cv2.VideoCapture('tabs.mp4') 
path="filter_sun.jpg"
size=(1280,720)
white=cv2.imread("whiite.jpg")
white=cv2.resize(white,size)
img = cv2.imread(path) 

img=cv2.resize(img,size)
fps=25


#writer = cv2.VideoWriter('o.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, size)


  
  
# loop runs if capturing has been initialized 
k=0
i=0
j=35
l=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(l)
jk=j
sw=False

#motion blur
size2=15
kernel_motion_blur = np.zeros((size2,size2))
kernel_motion_blur[int((size2-1)/2),:] = np.ones(size2)
kernel_motion_blur = kernel_motion_blur /  size2



kernel_identity = np.array([[10,0,-3,2],[2,0,-1,0],[-3,0,20,0],[0,2,1,-23]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
kernel_identity_2 = np.array([[15,13,-13],[0,11,0],[2,11,-2]])

#cv2.imshow('Identity filter',output)
while(i<l): 
  
    # reads frames from a camera 
    #while(i<l/10):
    ret, frame = cap.read()
     #   i+=1
  
    #ret, frame = cap.read()
    # converting BGR to HSV 
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
    # define range of red color in HSV 
    lower_red = np.array([30,10,0]) 
    upper_red = np.array([255,255,250]) 
    
    # create a red HSV colour boundary and  
    # threshold HSV image 
    # mask = cv2.inRange(hsv, lower_red, upper_red) 
  
    # Bitwise-AND mask and original image 
    # res = cv2.bitwise_and(frame,frame, mask= mask) 
  
    # Display an original image 
    # cv2.imshow('Original',frame) 
  
    # finds edges in the input image image and 
    # marks them in the output map edges 
    frame = cv2.resize(frame,size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(gray[0,0])
    inc=1
    edges = cv2.Canny(frame,jk,jk+inc)# inc is for stability
    #edges=1-edges
    #sys.stdout.write(str(i))
    #print(i)
    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    i+=1
    #res = cv2.GaussianBlur(frame,(55,55),cv2.BORDER_DEFAULT)
    res = cv2.bitwise_and(frame,frame,mask=edges)
    
    
    
    
    #print(res[0,0])
    #res= cv2.bitwise_or(res,cv2.GaussianBlur(frame,(47,47),cv2.BORDER_DEFAULT))
    #print(res)
    #break
    
    #res = cv2.GaussianBlur(res,(7,7),cv2.BORDER_DEFAULT)

     
    #output=cv2.filter2D(frame,-1,kernel_motion_blur)

    #res = cv2.bitwise_and(output,output,mask=edges)
    # Display edges in a frame 
    cv2.imshow('Edges',res) 
    #writer.write(res)
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(25) & 0xFF
    if k == 27: 
        break
  
  
# Close the window 
cap.release() 
print(dir(cv2))

#writer.release()
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  



