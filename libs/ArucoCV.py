import unittest

#besoin d'opencv-contrib-python
#class de detection aruco utilise egalement dans le noeud de suivi via aruco
#pip install opencv-contrib-python
import cv2
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

#generate aruco to follow 
class HarucoDetect:
    def __init__(self,aruco_dict=None ):
        if aruco_dict is None :
            #https://chev.me/arucogen/
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000) #aruco.DICT_6X6_250 aruco.DICT_6X6_100 aruco.DICT_6X6_1000 aruco.DICT_6X6_50)
        else:
            self.aruco_dict = aruco_dict

        #markerImage= cv2.aruco.generateImageMarker(self.aruco_dict, 23, 200, 1)
        #cv2.imwrite("marker23.png", markerImage);

        pass

    def processImage(self,image,arucoSize,mtx,dist,idList=[23]):

        print ("image.shape="+str(image.shape))
        if (len(image.shape)==2):
            gray=image.copy()  
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        idTxyzrpyList=[]#txyz=[0.0,0.0,0.0]
        #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=self.parameters)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, arucoSize, mtx, dist)
        print ("ids="+str(ids))
        if np.all(ids != None):
            for i in np.arange(len(ids)):
                print ("ids[i][0] =========="+str(ids[i][0] ))
                if ids[i][0] in idList:
                    logging.info(f'{ids[i][0]},  {tvec[i][0][0]}, {tvec[i][0][1]}, {tvec[i][0][2]}  ,  {rvec[i][0][0]}, {rvec[i][0][1]}, {rvec[i][0][2]}')

                    currentRvec=rvec[i][0][0:3]
                    currentTVec=tvec[i][0][0:3]
                    #print ("currentRvec="+str(currentRvec))
                    # ("currentTVec="+str(currentTVec))
                    Hrot=cv2.Rodrigues(currentRvec)[0]
                    #rpy = R.from_mat(Hrot).as_euler("ZYX",degrees=True).flip()
                    H=np.eye(4)
                    H[0,3]=currentTVec[0]
                    H[1,3]=currentTVec[1]
                    H[2,3]=currentTVec[2]
                    #print ("Hrot="+str(Hrot))
                    #print ("Hrot.shape="+str(Hrot.shape))
                    H[0:3,0:3]=Hrot
                    #print ("H="+str(H))
                    HcamToWorld=np.eye(4)
                    #HcamToWorld[0,1]=-1#x de vient -y
                    #HcamToWorld[1,2]=-1#y de vient -z
                    #HcamToWorld[2,0]=1#z de vient x

                    #HcamToWorld[0,0:3]=[0,-1,0] 
                    #HcamToWorld[1,0:3]=[0,0,-1] 
                    #HcamToWorld[2,0:3]=[1,0,0] 
                    #x de vient z

                    #print ("HcamToWorld="+str(HcamToWorld))
                    #Hworld=np.linalg.inv(HcamToWorld).dot(H)
                    #print ("Hworld="+str(Hworld))

                    HcamToWorld[0:3,0]=[0,-1,0] #x de vient -y
                    HcamToWorld[0:3,1]=[0,0,-1] #y de vient -z
                    HcamToWorld[0:3,2]=[1,0,0] #z de vient x
                    Hworld=HcamToWorld.dot(H)

                    txyz=Hworld[0:3,3]
                    rpy = np.flip(R.from_matrix(Hworld[0:3,0:3]).as_euler("ZYX",degrees=True))
                    #print ("rvec:"+str(rvec))
                    x=int(txyz[0]*100)/100.0
                    y=int(txyz[1]*100)/100.0
                    z=int(txyz[2]*100)/100.0
                    rx=int(rpy[0]*10)/10.0
                    ry=int(rpy[1]*10)/10.0
                    rz=int(rpy[2]*10)/10.0

                    idTxyzrpy=[ids[i][0],x,y,z,rx,ry,rz]
                    idTxyzrpyList.append(idTxyzrpy)
                    msg1="x:"+str(x)+",y:"+str(y)+",z:"+str(z)
                    msg2="roll:"+str(rx)+",pitch:"+str(ry)+",yaw:"+str(rz)
                    print (msg1)
                    print (msg2)
                    cv2.putText(img=image, text=msg1, org=(2, int(image.shape[1]/4)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
                    cv2.putText(img=image, text=msg2, org=(2, int(image.shape[1]/2)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)

                    try:
                        cv2.drawFrameAxes(image, mtx, dist, rvec, tvec, 0.1);
                    except:
                        pass
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)#affichage de l'aruco dans l'image

        return image,idTxyzrpyList
    
class ArucoTest(unittest.TestCase):
    def test_instanciate(self):
        harucoDetect = HarucoDetect()

    def test_detect(self):

        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        harucoDetect = HarucoDetect(aruco_dict)
        #fake mtx and dist
        mtx = np.array([[638.46300278,           0., 337.7732507 ],
                 [          0., 637.68233074, 277.09219809],
                 [          0.,           0.,           1.]])	   
        dist = np.array([[ 0.1128955 ,  0.38226535,  0.01402137,  0.00912032, -1.51235606]])
        arucoSize = 0.20 #0.20

        #create an image with aruco
        blank_image = np.zeros((640,480), np.uint8)
        blank_image[:,:]=255
        markerImage23= cv2.aruco.generateImageMarker(aruco_dict, 23, 200, 1)
        blank_image[100:300,100:300]=markerImage23
        image,distancedxyzrpy = harucoDetect.processImage(blank_image,arucoSize,mtx,dist, idList=[23,28])
        print ("distancedxyzrpy="+str(distancedxyzrpy))
        self.assertEqual(distancedxyzrpy[0][0], 23)#check detection

        markerImage28= cv2.aruco.generateImageMarker(aruco_dict, 28, 200, 1)
        blank_image[100:300,100:300]=markerImage28
        image,distancedxyzrpy = harucoDetect.processImage(blank_image,arucoSize,mtx,dist, idList=[23,28])
        self.assertEqual(distancedxyzrpy[0][0], 29)#check detection


if __name__ =='__main__':
    webcam = cv2.VideoCapture(0)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = webcam.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = webcam.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    cascadefile = "./haarcascade_frontalface_alt.xml"
    roiRatioX=0.2
    roiRatioY=0.2
    harucoDetect = HarucoDetect()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('./outputAruco.avi', fourcc, fps, (640,480))
    out=None

    arucoSize = 0.20 #0.20
    mtx = np.array([[638.46300278,           0., 337.7732507 ],
                 [          0., 637.68233074, 277.09219809],
                 [          0.,           0.,           1.]])	   
    dist = np.array([[ 0.1128955 ,  0.38226535,  0.01402137,  0.00912032, -1.51235606]])
    while webcam.isOpened():
        (result, image) = webcam.read()
        if result :
            if out is not None :
                out.write(image)
            image,distanceXYZ = harucoDetect.processImage(image,arucoSize,mtx,dist)
            
            # plt.imshow(image)
            cv2.imshow('OpenCV', image)
            key = cv2.waitKey(10)
            
    webcam.release()
    if out is not None :
        out.release()

    cv2.destroyAllWindows()