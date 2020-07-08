
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D




normal_dir = os.path.join('dataset/train/NORMAL')
pneumonia_dir = os.path.join('dataset/train/PNEUMONIA')
covid_dir = os.path.join('dataset/train/COVID')




print('total training normal images:', len(os.listdir(normal_dir)))
print('total training pneumonia images:', len(os.listdir(pneumonia_dir)))
print('total training covid images:', len(os.listdir(normal_dir)))

normal_files=os.listdir(normal_dir)
print(normal_files[:10])
pneumonia_files=os.listdir(pneumonia_dir)
print(pneumonia_files[:10])
covid_files=os.listdir(covid_dir)
print(covid_files[:10])

'''
pic_index = 2

next_normal = [os.path.join(normal_dir, fname)
                for fname in normal_files[pic_index-2:pic_index]]
next_pneumonia = [os.path.join(pneumonia_dir, fname)
                for fname in pneumonia_files[pic_index-2:pic_index]]
next_covid = [os.path.join(covid_dir, fname)
                for fname in covid_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_normal+next_pneumonia+next_covid):
    #print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
'''
TRAINING_DIR= 'dataset/train'
training_datagen = ImageDataGenerator(rescale = 1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

TEST_DIR = 'dataset/test'

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(216,216),
                                                       class_mode='categorical',
                                                       batch_size=28)

validation_generator = validation_datagen.flow_from_directory(TEST_DIR,
                                                              target_size=(216,216),
                                                              class_mode='categorical',
                                                              batch_size=28)






# Initialising the CNN
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(216, 216, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(312, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("covid_test.h5")