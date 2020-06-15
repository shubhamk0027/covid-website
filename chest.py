import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

labels=pd.read_csv('sample_labels.csv', dtype=str)

image_path={os.path.basename(x): x for x in glob(os.path.join('images/*png'))}

print('Images found:', len(image_path), 'Total headers:',labels.shape[0])

labels['path']=labels['Image Index'].map(image_path.get)


label_counts = labels['Finding Labels'].value_counts()[:15]
'''
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
'''
from itertools import chain
all_labels = np.unique(list(chain(*labels['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        labels[c_label] = labels['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
labels.head()

'''

from imblearn.over_sampling import SMOTE
smote=SMOTE('minority')
train_df, all_labels=smote.fit_sample(train_df, all_labels)
print(train_df, all_labels)

'''

labels['disease_vec'] = labels.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])






from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(labels,
                                   test_size = 0.25,
                                   random_state = 500,
                                   stratify = labels['Finding Labels'].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df.shape[0])




from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128,128)



datagen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range= 0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)






train_gen = datagen.flow_from_dataframe(dataframe= train_df,
                                        directory='./images/',
                                        x_col='Image Index',
                                        y_col = 'disease_vec',
                                        target_size = IMG_SIZE,
                                        color_mode = 'grayscale',
                                        class_mode='sparse',
                                        shuffle=True,
                                        batch_size = 32)

valid_gen = datagen.flow_from_dataframe(dataframe= valid_df,
                                        directory='./images/',
                                        x_col = 'Image Index',
                                        y_col = 'disease_vec',
                                        class_mode='sparse',
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 256) # we can use much larger batches for evaluation


import tensorflow as tf
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
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
    tf.keras.layers.Dense(15, activation='softmax')
])

model.summary()


model.compile(loss = 'categorical_cross_entropy', optimizer='adam', metrics=['accuracy'])
history= model.fit(train_gen,
                                  steps_per_epoch=10,
                                  validation_data = valid_gen,
                                  validation_steps=3,
                                  )
#history = model.fit(train_gen, epochs=25, validation_data = valid_gen, verbose = 1, validation_steps=3)

model.save("xray.h5")
