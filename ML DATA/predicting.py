from tensorflow.keras.models import load_model
import os
import numpy as np
import glob

from keras.preprocessing import image
loaded_model=load_model('covid_chest.h5')

uploaded=glob.glob('test/*.jpg')
disease_vec= ['ARDS', 'Atelectasis', 'Bacterial', 'COVID-19', 'Cardiomegaly', 'Chlamydophila', 'Consolidation', 'E.Coli', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'Mass', 'Mycoplasma Bacterial Pneumonia', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumocystis', 'Pneumonia', 'Pneumothorax', 'SARS', 'Streptococcus', 'Varicella']


for fn in uploaded:
    # predicting images
    path = fn
    img = image.load_img(path, color_mode='grayscale', target_size=(128, 128, 1))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = loaded_model.predict(images, batch_size=10)
    k = np.argmax(classes[0])
    print(disease_vec[k])