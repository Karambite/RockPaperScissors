import numpy as np
import cv2
import time
import tensorflow_hub as hub
import mediapipe as mp
import tensorflow as tf

FONT = cv2.FONT_HERSHEY_SIMPLEX


def makeInference(input_frame, model):
    predict_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    predict_frame = np.expand_dims(predict_frame, 0)
    predictions = model.predict(predict_frame)
    label = np.argmax(predictions)
    if label == 0:
        item = "rock"
    elif label == 1:
        item = "paper"
    else:
        item = "scissors"
    return item


def makeMove():
    randomArray = np.random.choice(3, 1)
    choice = randomArray[0]
    if choice == 0:
        item = "rock"
    elif choice == 1:
        item = "paper"
    else:
        item = "scissors"
    return item


def frameOutput(currentTime, startTime, width, height, frame):
    if currentTime - startTime >= 5:
        text = "SHOOT!"
    elif currentTime - startTime >= 3.5:
        text = "SCISSORS"
    elif currentTime - startTime >= 2:
        text = "PAPER"
    elif currentTime - startTime >= 0:
        text = "ROCK"
    textSize = cv2.getTextSize(text, FONT, 1, 2)[0]
    textX = int((width - textSize[0]) / 2)
    textY = int((height + textSize[1]) / 2)
    cv2.putText(frame, text, (textX, textY), FONT, 1, (0, 255, 0), 4)


def displayGuess(user, computer, frame):
    userDisplay = "YOU: " + user
    computerDisplay = "Computer: " + computer
    cv2.putText(frame, str(userDisplay), (10, 25), FONT, .7, (255, 0, 0), 2)
    cv2.putText(frame, str(computerDisplay), (10, 50), FONT, .7, (255, 0, 0), 2)


def defineWinner(user, computer, height, frame):
    if user == computer:
        result = "DRAW"
    elif user == "None":
        result = "USER WINS!"
    elif user == "rock" and computer == "scissors":
        result = "USER WINS!"
    elif user == "paper" and computer == "rock":
        result = "USER WINS!"
    elif user == "scissors" and computer == "paper":
        result = "USER WINS!"
    elif user == "None":
        result = "COMPUTER WINS!"
    elif computer == "rock" and user == "scissors":
        result = "COMPUTER WINS!"
    elif computer == "paper" and user == "rock":
        result = "COMPUTER WINS!"
    elif computer == "scissors" and user == "paper":
        result = "COMPUTER WINS!"
    result = "Winner: " + result
    cv2.putText(frame, result, (20, int(height - 10)), FONT, .7, (255, 0, 0), 2)


def getHand(frame, width, height):
    mp_hands = mp.solutions.hands
    hand_Images = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=.5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_Images.process(frame)

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
            new_frame = frame[int(y_min * width) - 50: int(x_min * height) - 50,
                        int(y_max * width) + 50:(int(x_max * height)) + 50]
            return new_frame, (y_min, x_min, y_max, x_max)
    else:
        return None, (None, None, None, None)


def main():
    model = tf.keras.models.load_model(
        "rockpaperscissorsmodel.h5",
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    startTime = time.time()
    vid = cv2.VideoCapture(0)
    while True:
        rec, frame = vid.read()
        width = vid.get(3)
        height = vid.get(4)
        currentTime = time.time()
        inference_frame, (y_min, x_min, y_max, x_max) = getHand(frame, width, height)
        if inference_frame and inference_frame.size > 0:
            cv2.rectangle(frame, (int(x_min * height), int(y_min * width)), (int(x_max * height), int(y_max * width)),
                          (0, 255, 0), 2)
            inference_frame = cv2.resize(frame, (150, 150))
            user = makeInference(inference_frame, model)
        else:
            user = "None"
        computer = makeMove()
        frameOutput(currentTime, startTime, width, height, frame)
        displayGuess(user, computer, frame)
        defineWinner(user, computer, height, frame)
        cv2.imshow("frame", frame)
        if currentTime - startTime > 6:
            time.sleep(3)
            startTime = time.time()
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()
