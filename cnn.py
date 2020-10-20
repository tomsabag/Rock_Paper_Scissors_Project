import tensorflow as tf
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Activation, Dropout
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix

image_gen = ImageDataGenerator(rotation_range=30, horizontal_flip=True, width_shift_range=0.1,
                               rescale=1/255, shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_image_gen = image_gen.flow_from_directory(directory='C:/Users/tomsa/PycharmProjects/OpenCV_Rock_Paper_Scissors/train',
                                                target_size=(150, 150), batch_size=16, class_mode='categorical')

test_image_gen = image_gen.flow_from_directory(directory='C:/Users/tomsa/PycharmProjects/OpenCV_Rock_Paper_Scissors/test',
                                                target_size=(150, 150), batch_size=16, class_mode='categorical')
# train_image_gen.class_indices : paper_train:0, rock_train:1, scissors_train:2

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

model.fit_generator(train_image_gen, epochs=5, validation_data=test_image_gen)
model.save_weights('weights_tom.h5')
print('saved succesfully')
#print(model.summary())
#model.load_weights('weights_tom.h5')
#print('here')
#print(model.evaluate_generator(test_image_gen, 50), model.metrics_names)
'''
count = 0

for i in range(1, 100):
    img = cv2.imread(f'test/scissors_test/scissors ({i}).png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    if model.predict_classes([[img]]) == [2]:
        count += 1
print(count)
plt.show()'''

