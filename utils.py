import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Dropout


def draw_rects(frame):
    cv2.rectangle(frame, (50, 75), (275, 300), (255, 255, 0), 3)
    cv2.rectangle(frame, (365, 75), (590, 300), (0, 255, 255), 3)

    cv2.putText(frame, 'Your move is:', (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, 'Computers move is:', (355, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def photo_button(frame, photo_taken):
    cv2.rectangle(frame, (268 + 75, 343), (397 + 75, 412), (0, 0, 0), -1)
    cv2.rectangle(frame, (270 + 75, 345), (395 + 75, 410), (0, 0, 255), -1)
    if not photo_taken:
        cv2.putText(frame, 'Play!', (310 + 75, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        cv2.putText(frame, 'Play again!', (275 + 75, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)


def clear_button(frame, already_played):
    cv2.rectangle(frame, (268 - 100, 343), (397 - 100, 412), (0, 0, 0), -1)
    cv2.rectangle(frame, (270 - 100, 345), (395 - 100, 410), (0, 255, 0), -1)

    cv2.putText(frame, 'CLEAR', (310 - 110, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def create_database_button(frame):
    cv2.rectangle(frame, (253, 420), (390, 474), (0, 0, 0), -1)
    cv2.rectangle(frame, (255, 422), (388, 472), (255, 0, 0), -1)

    cv2.putText(frame, 'Create Database', (255, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)



def build_model():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(units=512, activation='relu'))

    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights('weights_tom.h5')
    return model


def detect_move(frame):
    model = build_model()
    img = cv2.imread('Player_Images/play.jpg')
    img = cv2.resize(img, (150, 150))
    result = model.predict_classes([[img]])
    print(result)
    if result == 0:
        return 'Paper'
    elif result == 1:
        return 'Rock'
    elif result == 2:
        return 'Scissors'
    else:
        return 'ERROR!'


def rps_gui(frame, t, t_clicked, move):
    lst = ['rock.png', 'paper.png', 'scissors.png', 'question_mark.png']
    if t - t_clicked < 4:
        delta = t - t_clicked
        if 3 - (int(delta)) != 0:
            cv2.putText(frame, str(3 - (int(delta))), (470, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        im = cv2.imread(f'pictures/{lst[int(delta)]}')
        im = cv2.resize(im, (225, 225))
        frame[75:300, 365:590] = im
    else:
        photo_button_clicked = False

        if move is not None:
            im = cv2.imread(f'pictures/{move}.png')
            im = cv2.resize(im, (225, 225))
            frame[75:300, 365:590] = im


def draw_winning_move(frame, move):
    cv2.putText(frame, move, (200, 65), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)


def take_a_photo(frame):
    cv2.imwrite('C:/Users/tomsa/PycharmProjects/OpenCV_Rock_Paper_Scissors/Player_Images/play.jpg',
                frame[75:300, 50:275])


def draw_taken_photo(frame, photo_taken, photo_cleared):
    if photo_taken and not photo_cleared:
        frame[75:300, 50:275] = cv2.imread('Player_images/play.jpg')
    return frame


def declare_winner(move, computer_move):
    output_string = ''

    if move == computer_move:
        output_string = 'Tie!'
    elif (move == 'Rock' and computer_move == 'Scissors') or (move == 'Scissors' and computer_move == 'Paper')\
            or (move == 'Paper' and computer_move == 'Rock'):
        output_string = 'You win !'
    else:
        output_string = 'You lose ...'
    return output_string


def database_photos(frame, fps):
    lst = ['rock', 'paper', 'scissors']
    lst2 = ['happy', 'sad']
    cv2.putText(frame, f'Taking {lst[fps//50]} photo number:' + str(fps),
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(f'C:/Users/tomsa/PycharmProjects/OpenCV_Rock_Paper_Scissors/happy/img{fps - 1}.jpg',
                frame[75:300, 50:275])
