import cv2
import numpy as np
import os
import sys
import tensorflow as tf
# from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 100
IMG_HEIGHT = 100
NUM_CATEGORIES = 2
TEST_SIZE = 0.2

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels) # one hot encoding

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size = TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor= 'val_loss', verbose= 0, save_best_only= True, mode= 'auto')
    model.fit(x_train,y_train, epochs = EPOCHS, callbacks = [checkpoint], validation_split= 0.2)
    
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose = 2)

def load_data(data_dir):
    images = []
    labels = []
    final_size = (IMG_WIDTH, IMG_HEIGHT)

    for i in range(NUM_CATEGORIES):
        dir_name = os.path.join(data_dir, str(i))
        for files in os.listdir(dir_name):
            image = cv2.imread(os.path.join(dir_name, files))
            gImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resize_image = cv2.resize(gImage, final_size)
            images.append(resize_image)
            labels.append(i)

    images = np.array(images) / 255
    images = np.array(images).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) 
    labels = np.array(labels)      
    # print(images.shape)
    # print(labels.shape)
    return images, labels


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(200, (3, 3), activation= 'relu',
                               input_shape= (IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(pool_size= (2, 2)),

        tf.keras.layers.Conv2D(100, (3, 3), activation= 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size= (2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation= "relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation= "softmax")
    ])

    model.compile(
        optimizer= "adam",
        loss= "categorical_crossentropy", 
        metrics= ["accuracy"])
    return model

if __name__ == "__main__":
    main()
