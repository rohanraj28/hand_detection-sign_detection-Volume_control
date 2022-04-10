import cv2 as cv
from cv2 import FONT_HERSHEY_COMPLEX
from cv2 import COLOR_RGB2BGR
from cv2 import COLOR_BGR2RGB
import mediapipe as mp
#create a variable for hands in mp solutions module
# False means that the input is not static(webcam) 
hands = mp.solutions.hands.Hands(False,2,1,0.5,0.5)
#create a mp draw to draw
mpDraw = mp.solutions.drawing_utils
#to start the webcam 
capture = cv.VideoCapture(0)
capture.set(3,1440)
while True:
    isTrue, frame = capture.read()
    frame = cv.flip(frame,1)
    #height = 480,width = 640
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
   
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
                        myHand.append((Landmark.x,Landmark.y))
                        myHandx.append(Landmark.x)
                        myHandy.append(Landmark.y)
                    fingertip_ids = [myHandy[8],myHandy[12],myHandy[16],myHandy[20]]
                    thumb_tip = myHandx[4]
                    thumb_mcp = myHandx[2]
                    for ids in fingertip_ids:
                        mcp = myHandy.index(ids) - 2
                        if ids < myHandy[mcp]:
                            count[hand_label.upper()] += 1
                    if (hand_label == "Right" and thumb_tip<thumb_mcp):
                        count["RIGHT"] += 1
                    if (hand_label == "Left" and thumb_tip>thumb_mcp):
                        count["LEFT"] += 1
                cv.putText(frame,str(sum(count.values())),(100,300),cv.FONT_HERSHEY_SIMPLEX,5,(255,0,0),2)
    
    cv.imshow("Video",frame)
    
    if cv.waitKey(20) & 0xFF ==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

    
        

        
    
    
