from ctypes import POINTER, cast
from turtle import width
from cv2 import VideoCapture
import mediapipe as mp
import cv2 as cv
import numpy as np 
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL


#create a variable for hands in mp solutions module
# False means that the input is not static(webcam) 
hands = mp.solutions.hands.Hands(False,2,1,0.5,0.5)
#create a mp draw to draw
mpDraw = mp.solutions.drawing_utils
#to start the webcam 
capture = cv.VideoCapture(0)
capture.set(3,1440)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume  =cast(interface,POINTER(IAudioEndpointVolume))
volbar = 400
volper = 0

volMin,volMax = volume.GetVolumeRange()[:2]


while True:
    isTrue, frame = capture.read()
    frame = cv.flip(frame,1)
    
    #height = 480,width = 640
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    h,w,_ = frame.shape
    #show results
    results = hands.process(frameRGB)
    # if any hand is detected
    if results.multi_hand_landmarks != None :
                count = {"RIGHT":0, "LEFT":0}
                for hand_index,hand_info in enumerate(results.multi_handedness):
                    hand_label= hand_info.classification[0].label
                
                    handlandmarks = results.multi_hand_landmarks[hand_index]
            #multi handedness contains only label
                    
                    
                    myHandx = []
                    myHand = []
                    myHandy = []
                    mpDraw.draw_landmarks(frame,handlandmarks,mp.solutions.hands.HAND_CONNECTIONS)
                    for Landmark in handlandmarks.landmark:
                        myHand.append([int(Landmark.x*w),int(Landmark.y*h)]   )
                        myHandx.append(Landmark.x)
                        myHandy.append(Landmark.y)
                    if myHand != []:
                        x1,y1 = myHand[4][0],myHand[4][1] #thumb
                        x2,y2  = myHand[8][0],myHand[8][1] #index
                        cv.circle(frame,(x1,y1),13,(255,0,0),cv.FILLED)
                        cv.circle(frame,(x2,y2),13,(255,0,0),cv.FILLED)
                        cv.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
                        length = hypot(x2-x1,y2-y1)

                        vol = np.interp(length,[30,350],[volMin,volMax])
                        volbar=np.interp(length,[30,350],[400,150])
                        volper=np.interp(length,[30,350],[0,100])
                        volume.SetMasterVolumeLevel(vol, None)
                        cv.rectangle(frame,(50,150),(85,400),(0,0,255),4) # vid ,initial position ,ending position ,rgb ,thickness
                        cv.rectangle(frame,(50,int(volbar)),(85,400),(0,0,255),cv.FILLED)
                        cv.putText(frame,f"{int(volper)}%",(10,40),cv.FONT_ITALIC,1,(0, 255, 98),3)
                    
    cv.imshow("Video",frame)
    
    if cv.waitKey(20) & 0xFF ==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
