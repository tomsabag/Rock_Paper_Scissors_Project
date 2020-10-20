import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import tensorflow
# from keras.models import Sequential
# from keras.layers import Dense, MaxPool2D, Conv2D
import utils


cap = cv2.VideoCapture(0)

fps = 0
fps_database_clicked = -1
database_button_was_clicked = False
t_database = -1


width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

photo_button_clicked = False
t_clicked = - 1
computer_move = ''
photo_taken = False
photo_cleared = False
predicted = False
predicted_value = - 1


def buttons(event, x, y, params, args):
    global photo_button_clicked
    global t_clicked
    global computer_move
    global photo_taken
    global photo_cleared
    global predicted
    global fps
    global fps_database_clicked
    global t_database

    # Play Button
    if event == cv2.EVENT_LBUTTONDOWN and 268 + 75 < x < 397 + 75 and 343 < y < 412:
        photo_button_clicked = True
        photo_cleared = False
        predicted = False
        t_clicked = time.perf_counter()

        computer_move = random.choice(['Rock', 'Paper', 'Scissors'])

        utils.take_a_photo(frame)
        photo_taken = True

    # Clear Button
    if event == cv2.EVENT_LBUTTONDOWN and 268 - 100 < x < 397 - 100 and 343 < y < 412:
        photo_cleared = True
        frame[75:300, 50:275] = frame_copy[75:300, 50:275]
        print('clear')

    if event == cv2.EVENT_LBUTTONDOWN and 253 < x < 390 and 420 < y < 474:
        print('creating database')
        cv2.imwrite('C:/Users/tomsa/PycharmProjects/OpenCV_Rock_Paper_Scissors/database_train/img1.jpg',
                    frame[75:300, 50:275])
        database_button_was_clicked = True
        fps_database_clicked = fps
        t_database = time.perf_counter()

cv2.namedWindow('Rock Paper Scissors!')
cv2.setMouseCallback('Rock Paper Scissors!', buttons)

while True:
    ret, frame = cap.read()

    fps += 1

    t = time.perf_counter()

    frame = utils.draw_rects(frame)
    if photo_taken and not photo_cleared and predicted is False:
        predicted_value = utils.detect_move(frame)
        print(predicted_value)
        predicted = True

    frame_copy = frame.copy()

    utils.photo_button(frame, photo_taken)
    utils.clear_button(frame, photo_taken)
    utils.create_database_button(frame)

    frame = utils.draw_taken_photo(frame, photo_taken, photo_cleared)
    #frame = utils.rps_gui(frame, photo_button_clicked)

    if photo_button_clicked:
        utils.rps_gui(frame, t, t_clicked, computer_move)

    if photo_button_clicked and not photo_cleared:
        utils.draw_winning_move(frame, predicted_value)

    if t - t_clicked > 4 and computer_move != '':
        cv2.putText(frame, computer_move + '!', (555, 65), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        if predicted_value != '' and not photo_cleared:
            cv2.putText(frame, utils.declare_winner(predicted_value, computer_move),
                        (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    if fps - fps_database_clicked < 100 and fps_database_clicked != -1:
        if t - t_database < 3:
            fps_database_clicked += 1
            cv2.putText(frame, 'Starting in: ' + str(np.around(3 - (t - t_database), decimals=2)), (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            if (fps - fps_database_clicked) % 49 == 0 and (fps - fps_database_clicked != 98):
                t_database = t
            else:
                utils.database_photos(frame, fps - fps_database_clicked)
        if fps - fps_database_clicked == 99:
            fps_database_clicked = - 1



    cv2.imshow('Rock Paper Scissors!', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
