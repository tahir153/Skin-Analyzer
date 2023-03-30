import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, 

modal = tf.keras.models.load_model("/content/drive/MyDrive/dermnet/SkinDiseasePrediction.h5")

img = load_img("/content/drive/MyDrive/dermnet/SkinTest/Acne and Rosacea.jpg", target_size = (224, 224,3))
img = img_to_array(img)
img=np.asarray(img)
img_batch = np.expand_dims(img, axis=0)
img = img/255

skin_classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos']

pred_index = np.argmax(modal.predict(img_batch))
pred_class=skin_classes[pred_index]
print("\nThis patient has '{}' disease.".format(pred_class))