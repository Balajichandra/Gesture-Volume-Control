import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            #for hand landmark detection and drawing
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mphands.HAND_CONNECTIONS)
        return img
    def findposition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[handNo]
            #for detection of position of hands i,e we are id and location
            for id,lm in enumerate(myhands.landmark):
            #height width channel
                h,w,c = img.shape 
                # finding the position
                cx,cy = int(lm.x*w),int(lm.y*h) 
                # id is for detecting landmark
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                   cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmList
    
    
def main():  
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img) 
        lmList = detector.findposition(img) 
        if len(lmList) != 0:
            print(lmList[4]) 
        #FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        #display fps
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()    

