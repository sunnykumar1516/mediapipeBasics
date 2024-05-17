import cv2 as cv
import mediapipe as mp
import  numpy as np

mp_pose = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_pose.Hands()
def combine_edges_and_handTracking():
    edges = None
   
    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        orFrame = frame.copy()
        height, width, channel = frame.shape
        blank_img = np.zeros((height, width, channel), np.uint8)
        
        frame = cv.blur(frame, (5, 5))
        edges = cv.Canny(frame, 90, 120)
        
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        mask = orFrame[0:720, 900:1280]
        aoi = edges[0:720, 900:1280]
        frame[0:720, 900:1280] = aoi
        
        results = hands.process(cv.cvtColor(mask, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                positions = []
                for index, item in enumerate(hand.landmark):
                    h, w, _ = frame.shape
                    positions.append([int(item.x * w), int(item.y * h), item.z])
                    print(int(item.x * w), int(item.y * h), item.z)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    positions.append([int(item.x * w), int(item.y * h), item.z])
                    print(int(item.x * w), int(item.y * h), item.z)
                    
                    mp_drawing.draw_landmarks(aoi,
                                              hand_landmarks,
                                              connections=mp_hands.HAND_CONNECTIONS)
            
            frame[0:720, 900:1280] = aoi
            index = (positions[8][0], positions[8][1])
            centre = (positions[9][0], positions[9][1])
            
            
            
        
        cv.imshow("Image", frame)
        if cv.waitKey(1) == ord("q"):
            break


combine_edges_and_handTracking()