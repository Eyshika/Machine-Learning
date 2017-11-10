import cv2
import numpy as np
from imutils import face_utils
import argparse
import imutils
import dlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Demonstration of Thin-Plate-Spline Warping 
#
#
#
# Output: 
# Display of warped image
#
# Example:
# tpsWarpDemo('..\data\0505_02.jpg','map.mat','tpsDemoLandmark.mat')
#
#1. read two face images 
#2. apply Dlib facial landmarks detection 
#3. use landmarks set of points for face one, and landmarks set of face2 to warp face one to look like face two. 
#Iit is a good example to illustrate if your TPS is working well or not. 
#Test it on at least 10 different pair of faces4

def rect_to_bb(rect):
      #take a boundary predicted by dlib
      #and convert it to the format (x,y,w,h) as we
      #would normally do with OpenCV
      x=rect.left()
      y=rect.top()
      w=rect.right()-x
      h=rect.bottom()-y
      
      #return a tuple of (x,y,w,h)
      return (x,y,w,h)

def shape_to_np(shape, dtype="int"):
      #initialize the list of x,y coordinates
      coords= np.zeros((68,2), dtype=dtype)
      #loop over the 68 facial landmarks and convertthem
      # to a 2-tuple of x,y cordinates
      for i in range(0,68):
            coords[i]=(shape.part(i).x, shape.part(i).y)
      #eturn list of x,y coordinates
      return coords
def computeWl(x_1, y_1, NPs):
      rXp=[[i]*len(x_1) for i in x_1]
      
      rYp=[[i]*len(y_1) for i in y_1]
      rXp_tran=[[row[i] for row in rXp] for i in range(len(rXp))]
      rYp_tran=[[row[i] for row in rYp] for i in range(len(rYp))]
      wR=np.sqrt(np.power(np.subtract(rXp,rXp_tran),2)+np.power(np.subtract(rYp,rYp_tran),2))
      wK=radialBasis(wR)
      a=np.ones((NPs,1))
      wP=np.vstack([np.asarray(a).T,x_1,y_1]).T #transpose
      temp=np.vstack([np.asarray(wK).T,np.asarray(wP).T]).T
   
      wP_trans=np.transpose(wP)
      temp1=np.vstack([np.asarray(wP_trans).T,np.asarray(np.zeros((3,3))).T]).T
   
      
      wL=np.vstack([np.asarray(temp),np.asarray(temp1)])      
      return wL

def radialBasis(wR):
      rli=wR
      rli[np.where(wR==0)]=np.finfo(np.double).tiny
      temp=2*np.power(wR,2)
      ko=np.multiply(temp,np.log(rli))
      
      return ko


def tpsMap(wW, imgH, imgW, x_1, y_1, NPs):
      [X, Y]=np.mgrid[0:imgH,0:imgW]
#      X=[item for list in X for item in list]
      X=np.ravel(X)
      Y=np.ravel(Y)
#      Y=[item for list in Y for item in list]
      NWs=len(X)
      print(X)
      rx=[]
      wW_1=[]
      wW_2=[]
      rY=[]
      for i in range(NPs):
            rx.append(X)
            rY.append(Y)
      rxp=[[i]*NWs for i in x_1]
      
      ryp=[[i]*NWs for i in y_1]
      
      wR=np.sqrt(np.power(np.subtract(rxp,rx),2)+np.power(np.subtract(ryp,rY),2))
      
      wK=radialBasis(wR)
      wP=np.transpose(np.vstack([np.asarray(np.ones((NWs,1))).T, np.asarray(X).T,np.asarray(Y).T]).T)
      wL=np.transpose(np.vstack([np.asarray(wK),np.asarray(wP)]))
      
      for i in range(len(wW)):
            wW_1.append(wW[i][0])
            wW_2.append(wW[i][1])
      Xw=np.dot(wL,wW_1)
      Yw=np.dot(wL,wW_2)
      return Xw,Yw

def interp2d(X,Y, Image1, Xw, Yw, outH, outW):
      color=Image1.shape[2]
      imgwr=np.zeros((outH,outW,color))
      imgH=Image1.shape[0]
      imgW=Image1.shape[1]
      
      #rounding
      Xwi=np.round(Xw)
      Ywi=np.round(Yw)
      
      #bounding
      Xwi=np.array(np.maximum(np.minimum(Xwi,outH-1),0))
      Ywi=np.array(np.maximum(np.minimum(Ywi,outW-1),0))
      Xwi = map(int, Xwi)
      Ywi = map(int, Ywi)


      arr=np.array([Xwi,Ywi],np.int32)
      arr1=np.array([X,Y],np.int32)
     
      #convert 2d cordinated into 1 d 

      fiw=np.ravel_multi_index(arr,(outH, outW))
      fip=np.ravel_multi_index(arr1,(imgH,imgW))
      
      #warped image construction
      o_r=np.ravel(np.zeros((outH,outW)))
      for i in range(color):
            img_r=np.ravel(Image1[:,:,i])
            o_r[fiw]=img_r[fip]
            imgwr[:,:,i]=np.reshape(o_r,(outH,outW))
                   
      
      #filling of holes
      
      out=imgwr
      map1=np.ravel(np.zeros((outH,outW)))
      map1[fiw]=1 #mask
   #   map1=[[int(y) for y in x] for x in map1]
      map1=np.reshape(map1,(outH,outW))
      yi_arr, xi_arr=np.where(map1==0)
      print(yi_arr)
      if [yi_arr]:
            
            for ix in range(len(yi_arr)):
                  xi=xi_arr[ix]
                  yi=yi_arr[ix]
                
                  #find min windows which has non hole neighbors
                  nz=0
                  for h in range(1,4):
                        yixl=np.maximum(yi-h,0)
                        
                        yixu=np.minimum(yi+h,outH-1)
                        
                        xixl=np.maximum(xi-h,0)
                        
                        xixu=np.minimum(xi+h,outW-1)
                        
                        temp=map1[yixl:yixu,xixl:xixu]
                        ans=all(all(item==0 for item in items) for items in temp)
                        if ans==0:
                              
                              nz=1
                              break
                  #use median
                  if nz:
                        for colix in range(color):
                              win=imgwr[yixl:yixu,xixl:xixu,colix]
                              
                              out[yi,xi,colix]=np.median(win[np.where(map1[yixl:yixu, xixl:xixu]!=0)])
                              
#      # combine them into an output image0
      out=np.array(out,dtype=np.uint8)
      return out

#Image 1 will be warped with Image 2
Image1=cv2.imread("G:/stevens/algomus/2sned/ref/05.jpg")
Image2=cv2.imread("G:/stevens/algomus/2sned/tar/10.jpg")
#cv2.imshow('Image 1 original',Image1) #originL IMages
#cv2.imshow('Image2 original', Image2)
#getting size of images

path_shape_predictor="shape_predictor_68_face_landmarks.dat"
path_Image1="--G:/stevens/algomus/2sned/ref/05.jpg"
path_Image2="--G:/stevens/algomus/2sned/tar/10.jpg"

# construct the argument parser and parse the arguments
ap1=argparse.ArgumentParser()
ap2=argparse.ArgumentParser()


ap1.add_argument("-p","--shape_predictor_68_face_landmarks.dat",required=True,help="path to facial landmark predictor")
ap2.add_argument("-p","--shape_predictor_68_face_landmarks.dat",required=True,help="path to facial landmark predictor")

ap1.add_argument("-i",path_Image1,required=True,help="path to input image")
ap2.add_argument("-i",path_Image2,required=True,help="path to input image")

#args=vars(ap.parse_args())

#initialize dlib's face detector (HOG-based) and then create
#the facial landmark predictor

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(path_shape_predictor)
Image1=imutils.resize(Image1, width=500)
Image2=imutils.resize(Image2, width=500)

rects1=detector(Image1,1)
rects2=detector(Image2,1)

#loop over face detection
for ((i,rect1), (j, rect2)) in zip(enumerate(rects1),enumerate(rects2)):
      #determine the facial landmarks for the face region, then
      #convert the facial landmark x,y coordinate to a NumPy array
      shape1=predictor(Image1, rect1)
      shape1=face_utils.shape_to_np(shape1)
      shape2=predictor(Image2, rect2)
      shape2=face_utils.shape_to_np(shape2)
      #convert dlib's rectangle to a OpenCv style bounding box i.e. x,y,w,h then draw the face bounding box
      (x1,y1,w1,h1)=face_utils.rect_to_bb(rect1)
      (x2,y2,w2,h2)=face_utils.rect_to_bb(rect2)


      #loop over cordinates of facial landmark
      for ((x1,y1),(x2,y2)) in zip(shape1, shape2):
            cv2.circle(Image1, (x1,y1), 1, (0,0,255), -1)
            cv2.circle(Image2, (x2,y2), 1, (0,0,255), -1)



#size of image
imgH,imgW=Image1.shape[:2]
outH, outW=Image2.shape[:2]

x_1=[]
x_2=[]
y_1=[]
y_2=[]
#landmarks
for i in range(len(shape1)):
      x_1.append(shape1[i][1])
      y_1.append(shape1[i][0])
      x_2.append(shape2[i][1])
      y_2.append(shape2[i][0])
NPs=len(shape1)

#thin plate spline algebra
wL=computeWl(x_1, y_1, NPs)
zero=np.zeros((3,2))
temp=np.vstack([np.asarray(x_2).T,np.asarray(y_2).T]).T #transpose
wY=np.vstack([temp, zero])
wW=np.dot(np.linalg.inv(wL),wY) #multiplication

#thin plate spline mapping

Xw, Yw= tpsMap(wW, imgH, imgW, x_1, y_1, NPs)
# warping
[X, Y]=np.mgrid[0:imgH,0:imgW]
#nearest neighbor or inverse

result=interp2d(X,Y, Image1, Xw, Yw, outH, outW)
#show output image with the face detection +facial landmarks

cv2.imwrite('01.png',result)


b,g,r = cv2.split(Image1)       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb
b,g,r = cv2.split(result)       # get b,g,r
rgb_imgr = cv2.merge([r,g,b])     # switch it to rgb
b,g,r = cv2.split(Image2)       # get b,g,r
rgb_img2 = cv2.merge([r,g,b])     # switch it to rgb

fig=plt.figure()
a=fig.add_subplot(1,3,1)
a.imshow(rgb_img)
a.set_title('Refrenced')

a2=fig.add_subplot(1,3,2)
a2.imshow(rgb_imgr)
a2.set_title('Warped')

a3=fig.add_subplot(1,3,3)
a3.imshow(rgb_img2)
a3.set_title('User')

#fig.savefig("02.png")
#TPS Warping
#shape 1 landmarks of image1 and shape2 landmarks of image 2
#computing thin plate spline

cv2.waitKey(0)




