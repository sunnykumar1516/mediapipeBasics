import math
import numpy as np
import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_video():
    writer = None
    (W, H) = (None, None)

    path = "../videos/cricket/cricket.mp4"
    vs = cv.VideoCapture(path)
    
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        with mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5,
                model_complexity=2
                ) as pose:

            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            annotated_image = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("op2.mp4", fourcc, 24,
                                    (frame.shape[1], frame.shape[0]), True)
        print("writing")
        writer.write(annotated_image)

        cv.imshow('image', annotated_image)
        cv.waitKey(24)


process_video()