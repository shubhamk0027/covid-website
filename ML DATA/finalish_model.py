#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import os
from glob import glob
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('sample_labels.csv', dtype=str)
df.head()


# In[3]:


image_path={os.path.basename(x): x for x in glob(os.path.join('images/*png'))}


# In[4]:


print('Images found:', len(image_path), 'Total headers:',df.shape[0])


# In[5]:


df['path']=df['Image Index'].map(image_path.get)


# In[6]:


df.head()


# In[7]:


labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "COVID-19", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]


# In[8]:



df_counts = df['Finding Labels'].value_counts()[:17]


# In[9]:


fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(df_counts))+0.5, df_counts)
ax1.set_xticks(np.arange(len(df_counts))+0.5)
_ = ax1.set_xticklabels(df_counts.index, rotation = 90)


# In[10]:



from itertools import chain
all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)


# In[11]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(df,
                                   test_size = 0.10,
                                   random_state = 2500,
                                   stratify = df['Finding Labels'].map(lambda x: x[:4]))
print('train shape: ', train_df.shape[0], 'validation shape:', valid_df.shape[0])


# In[12]:


train_df.head()


# In[13]:


def check_for_leakage(df1, df2, patient_col):
    df1_patients_unique = set(df1[patient_col].unique().tolist())
    df2_patients_unique = set(df2[patient_col].unique().tolist())
    
    patients_in_both_groups = list(df1_patients_unique.intersection(df2_patients_unique))
    n_overlap = len(patients_in_both_groups)
    # leakage contains true if there is patient overlap, otherwise false.
    leakage = True if(len(patients_in_both_groups)>1) else False # boolean (true if there is at least 1 patient in both groups)
    
    if(leakage):
        print("There are", len(patients_in_both_groups), " patients in both groups.")
        
        train_overlap_idxs = []
        valid_overlap_idxs = []
        for idx in range(n_overlap):
            train_overlap_idxs.extend(train_df.index[train_df['Patient ID'] == patients_in_both_groups[idx]].tolist())
            valid_overlap_idxs.extend(valid_df.index[valid_df['Patient ID'] == patients_in_both_groups[idx]].tolist())
    
        train_df.drop(train_overlap_idxs, inplace=True)
    ### END CODE HERE ###
    
    return leakage


# In[14]:


print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'Patient ID')))


# In[15]:


print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'Patient ID')))


# In[16]:


from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (320,320)


# In[17]:


datagen = ImageDataGenerator(samplewise_center=True,
                             samplewise_std_normalization=True,
                             horizontal_flip = False,
                             vertical_flip = False,
                             rescale = 1./255,
                             height_shift_range= 0.05,
                             width_shift_range=0.1,
                             rotation_range=5,
                             shear_range = 0.1,
                             fill_mode = 'reflect',
                             zoom_range=0.15)


# In[18]:



train_gen = datagen.flow_from_dataframe(dataframe= train_df,
                                        directory='./images/',
                                        x_col='Image Index',
                                        y_col = labels,
                                        target_size = IMG_SIZE,
                                        class_mode='raw',
                                        shuffle=True,
                                        batch_size = 32)


# In[19]:


valid_gen = datagen.flow_from_dataframe(dataframe= valid_df,
                                        directory='./images/',
                                        x_col = 'Image Index',
                                        y_col = labels,
                                        class_mode='raw',
                            target_size = IMG_SIZE,
                            # color_mode = 'grayscale',
                            batch_size = 32)


# In[20]:


import tensorflow as tf


# In[21]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_gen.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# In[22]:


def compute_class_freqs(labels):
    
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0)/N
    negative_frequencies = 1- positive_frequencies

    return positive_frequencies, negative_frequencies


# In[23]:


freq_pos, freq_neg = compute_class_freqs(train_gen.labels)


# In[26]:


import seaborn as sns


# In[ ]:





# In[ ]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v}
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);

from keras import backend as K


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """

    def weighted_loss(y_true, y_pred):


        loss = 0.0



        for i in range(len(pos_weights)):

            loss += -(K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) + neg_weights[i] * (
                        1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon), axis=0))


        return loss



    return weighted_loss


# In[ ]:


from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
base_model = tf.keras.applications.MobileNetV2(input_shape = (320,320,3), include_top = False, weights = 'imagenet')
base_model.trainable=False
base_model.summary()



# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(all_labels), activation = 'sigmoid')


# In[ ]:


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# In[ ]:


model.compile(optimizer = 'adam', loss=get_weighted_loss(pos_weights, neg_weights),
                           metrics = ['accuracy', 'categorical_accuracy', 'binary_accuracy'])


model.summary()


history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=147,
                    validation_data=valid_gen,
                    validation_steps=18,
                              epochs=50
                              )


# In[ ]:


model.save('chest_fi.h5')


# In[ ]:


plt.figure(figsize=[8,6])
plt.plot(history.history['loss'], 'black', linewidth=3.0)
plt.plot(history.history['val_loss'], 'black',ls='--', linewidth=3.0)
plt.legend(["Training loss", "validatoin loss"], fontsize=18)
plt.xlabel("Epochs", fontsize=16)
plt.xlabel("Loss", fontsize=16)
plt.title("Loss Curve")


# In[ ]:


plt.figure(figsize=[8,6])
plt.plot(history.history['binary_accuracy'], 'black', linewidth=3.0)
plt.plot(history.history['val_binary_accuracy'], 'black',ls='--', linewidth=3.0)
plt.legend(["Training acc", "validation acc"], fontsize=18)
plt.xlabel("Epochs", fontsize=16)
plt.xlabel("Accuracy", fontsize=16)
plt.title("Accuracy Curve")


# In[ ]:




