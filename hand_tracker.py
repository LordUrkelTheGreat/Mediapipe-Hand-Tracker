# import libraries
import cv2
import mediapipe as mp
import os
import shutil

# remove output folder if it exists
shutil.rmtree("Output Images")

# create output folder to store results
os.mkdir("Output Images")

# frame counter
frame_counter = 0

# setting up mediapipe hand landmarks (each landmark per individual joint in hand)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# setup camera feed
camera = cv2.VideoCapture(1)    # I set it to 1 for my camera

# instantiating mediapipe hands model (if you want to track more than 2 hands, change max_num_hands value)
with mp_hands.Hands(model_complexity = 0, min_detection_confidence = 0.8, min_tracking_confidence = 0.5, max_num_hands = 10) as hands:
    # create an infinite loop for video
    while camera.isOpened():
        # read the data per frame
        data, frame = camera.read()

        # if there was no data captured from current frame
        if not data:
            print("Ignoring current (empty) camera frame.")
            continue

        # detections: threshold for the initial detection to be successful
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # change frame color from BGR to RGB for mediapipe
        image.flags.writeable = False                       # set the image's writeable flag to False to improve performance
        results = hands.process(image)                      # this is what makes the hands be detected
        image.flags.writeable = True                        # set the image's writeable flag to True to change the frame's color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # change frame color back from RGB to BGR

        # print frame's results (comment these out if it prints something)
        # multi_hand_landmarks:
        #   x = landmark position in the horizontal axis
        #   y = landmark position in the vertical axis
        #   z = landmark depth from the camera
        #print(results)
        #print(results.multi_hand_landmarks)

        # tracking: threshold for the tracking after the initial detection

        # render current frame results
        if results.multi_hand_landmarks:
            # go through each hand landmark found in the current frame
            for (num, hand) in enumerate(results.multi_hand_landmarks):
                # draw the landmarks onto the video
                #mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)      # comment the next line out if you want to use the default landmark color/size settings
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4),
                                          mp_drawing.DrawingSpec(color = (121, 44, 250), thickness = 2, circle_radius = 2),
                                          )
                
        # store the current frame's results into output folder
        cv2.imwrite(os.path.join("Output Images", "opencv_frame_{}.jpg".format(frame_counter)), image)

        # create video window with captured frame
        #cv2.imshow("Hand Tracker", image)      # comment this out and uncomment the next line if you want to flip the video
        cv2.imshow("Hand Tracker", cv2.flip(image, 1))

        # if user wants to exit the program
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            print("Exiting program.")
            break

        # increment the frame counter
        frame_counter = frame_counter + 1

# close the program
camera.release()
cv2.destroyAllWindows()