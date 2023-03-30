from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
# import tkinter as tk
# from tkinter import filedialog
# from tkinter import *

# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from keras.models import load_model
app = Flask(__name__)


model = load_model('SkinDiseasePrediction.h5')

def predict_label(img_path):
    img = image.load_img(img_path, target_size = (224, 224,3))
    img = image.img_to_array(img)
    img=np.asarray(img)
    img_batch = np.expand_dims(img, axis=0)
    img = img/255
    skin_classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos']
    pred_index = np.argmax(model.predict(img_batch))
    pred_class=skin_classes[pred_index]
    return pred_class

# Flask Code

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return """ABOUT Skin diseases are common but often difficult to diagnose accurately,
leading to delays in treatment and potentially serious health complications. Traditional 
diagnosis methods require specialized knowledge and may not be available in all areas, 
leading to disparities in healthcare access. An AI-powered skin disease detection system
could provide a faster, more accessible, and more accurate diagnosis, improving patient outcomes
and reducing the burden on healthcare systems."""

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = img.filename
		img_path = "static/" + img.filename
		img.save(img_path)
		pred_disease = predict_label(img_path)
	return render_template("index.html", predicted_class = pred_disease, img_path = img_path)

@app.route("/TestImages/<fname>", methods = ['GET'])
def get_img(fname):
    return send_file(f'TestImages/{fname}')

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)