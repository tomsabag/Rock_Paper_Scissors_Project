import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('test', frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        cv2.imwrite('C:/Users/tomsa/PycharmProjects/OpenCV_Rock_Paper_Scissors/Player_Images/img.jpg', frame)
        print('click')

    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
