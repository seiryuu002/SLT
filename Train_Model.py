import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import warnings
import numpy as np
import cv2
import os
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping

warnings.simplefilter(action='ignore', category=FutureWarning)

path = r'Model/'
train_path = r'F:\\SLT\\RealSLT\\Data\\train'
test_path = r'F:\\SLT\\RealSLT\\Data\\test'

train_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
test_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
train_batches = train_data.flow_from_directory(directory=train_path, target_size=(128, 128), class_mode='categorical', batch_size=10,shuffle=True)
test_batches = test_data.flow_from_directory(directory=test_path, target_size=(128, 128), class_mode='categorical', batch_size=10, shuffle=True)

images, labels = next(train_batches)


# Plotting the sample images...
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 30))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# To plot graph of learning rate and accuracy...
def plot_graph():
    plt.plot(history2.history['accuracy'], label='accuracy')
    plt.plot(history2.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


plot_images(images)
print(images.shape)
print(labels)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters = 128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(27,activation ="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop],  validation_data = test_batches)
images, labels = next(train_batches) # For getting next batch of images...

images, labels = next(test_batches) # For getting next batch of images...
scores = model.evaluate(images, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

if not os.path.exists(path):
    os.makedirs(path)
model.save(path + 'sign_detector_model.h5')

print(history2.history)

images, labels = next(test_batches)

model = keras.models.load_model(path + "sign_detector_model.h5")

scores = model.evaluate(images, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

model.summary()

scores
model.metrics_names


letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
           10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'Space',
           20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

predictions = model.predict(images, verbose=0)
print("predictions on a small set of test data--")
print("")
for ind, i in enumerate(predictions):
    print(letters[np.argmax(i)], end="    ")

plot_images(images)
print('Actual labels')
for i in labels:
    print(letters[np.argmax(i)], end="    ")
print(images.shape)
plot_graph()
history2.history
