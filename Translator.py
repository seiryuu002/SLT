import numpy as np
import cv2
import keras
import tensorflow as tf

model = keras.models.load_model(r"Model\\sign_detector_model_2_90.h5")
print("Model Loaded...")

letters = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I', 9 : 'J',
           10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R', 18 : 'S', 19 : 'Space',
           20: 'T', 21 : 'U',22 : 'V',23 : 'W',24 : 'X',25 : 'Y',26 : 'Z'}
clear = False
prediction = None
background = None
accumulated_weight = 0.6
word = ''
sentence = ''

roi_x = 400
roi_y = 50
roi_x1 = 600
roi_y1 = 250

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)


def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)


def message():
    global word, prediction, sentence

    k = cv2.waitKey(10) & 0xFF
    if k == 32:
        if letters[np.argmax(prediction)] != 'Space':
            word += letters[np.argmax(prediction)]
        else:
            word += ' '
    if word != '':   
        if word[-1] == ' ':
            sentence += word
            word = ''



def screen_events(event, x, y, flags, param):
    global clear ,word, sentence

    if (event == cv2.EVENT_LBUTTONDBLCLK) & (1240 < x < 1270) & (440 < y < 470):
        word = ''
        sentence = ''
    if (event == cv2.EVENT_LBUTTONDBLCLK) & (1240 < x < 1270) & (405 < y < 435):
        image = cv2.imread('F:\\SLT\\RealSLT\\Members\\Dragon.png')
        cv2.imshow("About This Project", image)
    if (event == cv2.EVENT_LBUTTONDBLCLK) & (1240 < x < 1270) & (370 < y < 400):
        cv2.waitKey(0)
        if event == cv2.EVENT_RBUTTONDBLCLK:
            pass



def screen():
    global prediction, i, clear, sentence
    cv2.rectangle(right_screen, (0, 0), (640, 480), (55, 55, 55), -1)
    cv2.rectangle(right_screen, (600,370), (630, 400), (200, 200, 200), -1)
    cv2.rectangle(right_screen, (600,405), (630, 435), (200, 200, 200), -1)
    cv2.rectangle(right_screen, (600,440), (630, 470), (200, 200, 200), -1)
    cv2.putText(right_screen, "| |", (605, 390 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    cv2.putText(right_screen, " i ", (602, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    cv2.putText(right_screen, "<-", (602, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    cv2.putText(right_screen, "SIGN LANGUAGE TRANSLATOR", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
    cv2.putText(right_screen, "Letter: " + letters[np.argmax(prediction)], (5, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
    cv2.putText(right_screen, "Word: " + word, (5, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
    cv2.putText(right_screen, "Message:" + sentence, (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)


cam = cv2.VideoCapture(0)
num_frames =0

cv2.namedWindow('Sign Detection')


while True:
    ret, frame = cam.read()
    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    right_screen = np.zeros((frame_copy.shape[0], frame_copy.shape[1], frame_copy.shape[2]), dtype=frame.dtype)
    # ROI from the frame
    roi = frame[roi_y : roi_y1, roi_x : roi_x1]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    k = cv2.waitKey(10) & 0xFF

    if num_frames <= 120:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)
    
    else: 
        # segmenting the hand region
        hand = segment_hand(gray_frame)
        

        # Checking if we are able to detect the hand...
        if hand is not None:
            
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_x, roi_y)], -1, red, 1)
            
            cv2.imshow("Thesholded Hand Image", thresholded)
            
            thresholded = cv2.resize(thresholded, (128, 128))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
            
            prediction = model.predict(thresholded)
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (roi_x, roi_y), (roi_x1, roi_y1), green, 2)
    # incrementing the number of frames for tracking
    num_frames += 1
    # Display the frame with segmented hand
    cv2.putText(frame_copy, "Recognizing Hand sign...", (10, 20), cv2.FONT_ITALIC, 0.5, red, 1)
    message()
    screen()
    cv2.setMouseCallback('Sign Detection', screen_events)
    final_screen = np.hstack((frame_copy, right_screen))
    cv2.imshow('Sign Detection', final_screen)

    # Close windows with Esc
    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()