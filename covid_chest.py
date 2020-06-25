import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

labels=pd.read_csv('sample_labels.csv', dtype=str)

image_path={os.path.basename(x): x for x in glob(os.path.join('images/*png'))}


print('Images found:', len(image_path), 'Total headers:',labels.shape[0])

labels['path']=labels['Image Index'].map(image_path.get)


#disease_vec = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]



label_counts = labels['Finding Labels'].value_counts()[:17]

labels.head()


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


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(labels,
                                   test_size = 0.25,
                                   random_state = 500,
                                   stratify = labels['Finding Labels'].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df.shape[0])



disease_vec= ['ARDS', 'Atelectasis', 'Bacterial', 'COVID-19', 'Cardiomegaly', 'Chlamydophila', 'Consolidation', 'E.Coli', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'Mass', 'Mycoplasma Bacterial Pneumonia', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumocystis', 'Pneumonia', 'Pneumothorax', 'SARS', 'Streptococcus', 'Varicella']

from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128,128)



datagen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip = True,
                              vertical_flip = False,
                             rescale = 1./255,
                              height_shift_range= 0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)






train_gen = datagen.flow_from_dataframe(dataframe= train_df,
                                        directory='./images/',
                                        x_col='Image Index',
                                        y_col = disease_vec,
                                        target_size = IMG_SIZE,
                                        color_mode = 'grayscale',
                                        class_mode='raw',
                                        shuffle=True,
                                        batch_size = 32)

valid_gen = datagen.flow_from_dataframe(dataframe= valid_df,
                                        directory='./images/',
                                        x_col = 'Image Index',
                                        y_col = disease_vec,
                                        class_mode='raw',
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 256)


import tensorflow as tf
'''
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')
])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
base_mobilenet_model = MobileNet(input_shape = (128,128,1), include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.summary()


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=3)
callbacks_list = [checkpoint, early]

history = multi_disease_model.fit_generator(generator=train_gen,
                    steps_per_epoch=10,
                    validation_data=valid_gen,
                    validation_steps=10,
                    epochs=1)


multi_disease_model.save('covid_chest.h5')
