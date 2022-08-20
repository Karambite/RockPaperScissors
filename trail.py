#from imutils import face_utils
import numpy as np
import mediapipe as mp
#import imutils
import cv2

vid = cv2.VideoCapture(0)
while True:
    rec, frame = vid.read()

    mpFaceMesh = mp.solutions.hands
    faceMeshImages = mpFaceMesh.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=.5)
    mp_drawing = mp.solutions.drawing_utils

    mp_drawing_styles = mp.solutions.drawing_styles

    image = frame  # cv2.imread("1361407883.jpeg")
    image = cv2.resize(image, (500,500))
    width, height, colors = image.shape
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = faceMeshImages.process(color)
    annotated_image = image.copy()
    img_h, img_w = frame.shape[:2]
    if results.multi_hand_landmarks:
        landmarks = np.array([[p.x, p.y] for p in results.multi_hand_landmarks[0].landmark])
        for face_landmarks in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = width
            y_min = height

            for i in landmarks:
                x, y = i[0], i[1]
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            #new_frame = annotated_image[int(y_min * width) -50: int(x_min * height)-50, int(y_max * width)+50:(int(x_max * height))+50]
            #new_frame = cv2.resize(new_frame, (500, 500))
            #cv2.imshow("new frame", new_frame)

            cv2.rectangle(annotated_image, (int(x_min * height), int(y_min * width)),
            (int(x_max * height), int(y_max * width)),
            (0, 255, 0), 2)
    else:
        cv2.putText(annotated_image, "NO FACE DETECTED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Output", annotated_image)
    key = cv2.waitKey(1)
    if key == 27:
        break
