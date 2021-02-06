import cv2
import numpy as np
import os
import string


path = 'Data_RGB/'
data_type = {0: 'Train', 1: 'Test'}
letters = {0: 'Space', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
           11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'R', 20: 'T',
           21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}
letter = 0
mode = 0
count = {}

roi_x = 350
roi_y = 50
roi_x1 = 550
roi_y1 = 250
green = (0, 255, 0)
red = (0, 0, 255)


def a(self):
    pass


cv2.namedWindow('Data Collector')
cv2.createTrackbar('Letter', 'Data Collector', 1, 26, a)
cv2.createTrackbar('Mode', 'Data Collector', 0, 1, a)


def get_track_pos():
    global letter, mode
    letter = cv2.getTrackbarPos('Letter', 'Data Collector')
    mode = cv2.getTrackbarPos('Mode', 'Data Collector')


def no_of_imgs():
    global count
    count = {'Space': len(os.listdir(path + data_type[mode]+"/Space")),
             'A': len(os.listdir(path + data_type[mode]+"/A")),
             'B': len(os.listdir(path + data_type[mode]+"/B")),
             'C': len(os.listdir(path + data_type[mode]+"/C")),
             'D': len(os.listdir(path + data_type[mode]+"/D")),
             'E': len(os.listdir(path + data_type[mode]+"/E")),
             'F': len(os.listdir(path + data_type[mode]+"/F")),
             'G': len(os.listdir(path + data_type[mode]+"/G")),
             'H': len(os.listdir(path + data_type[mode]+"/H")),
             'I': len(os.listdir(path + data_type[mode]+"/I")),
             'J': len(os.listdir(path + data_type[mode]+"/J")),
             'K': len(os.listdir(path + data_type[mode]+"/K")),
             'L': len(os.listdir(path + data_type[mode]+"/L")),
             'M': len(os.listdir(path + data_type[mode]+"/M")),
             'N': len(os.listdir(path + data_type[mode]+"/N")),
             'O': len(os.listdir(path + data_type[mode]+"/O")),
             'P': len(os.listdir(path + data_type[mode]+"/P")),
             'Q': len(os.listdir(path + data_type[mode]+"/Q")),
             'R': len(os.listdir(path + data_type[mode]+"/R")),
             'S': len(os.listdir(path + data_type[mode]+"/S")),
             'T': len(os.listdir(path + data_type[mode]+"/T")),
             'U': len(os.listdir(path + data_type[mode]+"/U")),
             'V': len(os.listdir(path + data_type[mode]+"/V")),
             'W': len(os.listdir(path + data_type[mode]+"/W")),
             'X': len(os.listdir(path + data_type[mode]+"/X")),
             'Y': len(os.listdir(path + data_type[mode]+"/Y")),
             'Z': len(os.listdir(path + data_type[mode]+"/Z"))}
    cv2.putText(left_screen, data_type[mode], (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, green, 1)
    cv2.putText(left_screen, "Space : " + str(count['Space']),
                (10, 35), cv2.FONT_HERSHEY_PLAIN, 1, green, 1)
    y_point = 50
    for i in string.ascii_uppercase:
        cv2.putText(left_screen, i + ": " + str(count[i]),
                    (10, y_point), cv2.FONT_HERSHEY_PLAIN, 1, green, 1)
        y_point += 15


def directory():
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + data_type[mode]):
        os.makedirs(path + data_type[mode])
    if not os.path.exists(path + data_type[mode] + '/Space'):
        os.makedirs(path + data_type[mode] + '/Space')
    for i in string.ascii_uppercase:
        if not os.path.exists(path + data_type[mode] + '/' + i):
            os.makedirs(path + data_type[mode] + '/' + i)


cap = cv2.VideoCapture(0)
num_frames = 0
while True:
    ret, frame = cap.read()
    # flipping image to simulate mirror image
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    left_screen = np.zeros((frame_copy.shape[0], 400, frame_copy.shape[2]), dtype=frame.dtype)
    # To get image just from ROI
    roi = frame[roi_y: roi_y1, roi_x: roi_x1]
    get_track_pos()
    directory()
    no_of_imgs()
    k = cv2.waitKey(10) & 0xFF
    if num_frames <= 100:
        cv2.putText(frame_copy, "Adjust hand...Gesture for " +
                    letters[letter], (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        cv2.putText(frame_copy, str(num_frames)+"/100 starting...",
                    (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        # Display the image from roi
        cv2.imshow("Hand Image", roi)
        num_frames += 1
    else:
        cv2.putText(frame_copy, "Adjust hand...Gesture for " +
                    letters[letter], (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        cv2.imshow("Hand Image", roi)
        if k == 32:
            cv2.imwrite(path + data_type[mode] + '/' + letters[letter] +
                        '/' + str(count[letters[letter]]) + '.jpg', roi)
            cv2.putText(left_screen, "images captured till now - " +
                        str(count[letters[letter]]), (10, 440), cv2.FONT_HERSHEY_PLAIN, 1, green, 1)
    # Drawing ROI on frame
    cv2.rectangle(frame_copy, (roi_x, roi_y), (roi_x1, roi_y1), green, 2)
    # Display final screen
    final_screen = np.hstack((left_screen, frame_copy))
    cv2.imshow('Data Collector', final_screen)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
