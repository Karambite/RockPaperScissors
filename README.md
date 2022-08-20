# RockPaperScissors
An interactive rock paper scissors game using openCV, mediaPipe, and Tensorflow

In order to start the game, first you must train the model (createModel.py) and save the model. This model will used to run inferences in rockpaperscissors.py.
MediaPipe is used for hand detection in the frame (easy to filter out other body parts or objects). OpenCV is used to access the webcam and display what it sees.
