import os
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import numpy as np
from tensorflow.keras.models import load_model

vgg19_model = load_model('./models/testvgg_19.h5')

arabic_model = load_model('./models/arabic_model.h5')
arabic_onehot_encoder = joblib.load("./models/arabic_encoder.pkl")


english_model = load_model('./models/english_model.h5')
english_onehot_encoder = joblib.load("./models/english_encoder.pkl")


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def file_is_allowed(filename: str):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)            
                
@app.route("/arabic", methods=['GET', 'POST'])
def arabic_classification():
    if request.method == "GET":
        return render_template("arabic_classification.html", show_result=False)
    else:
        
        if 'file' not in request.files:
            return "No file was uploaded"
        
        file = request.files['file']
        
        if file.filename == "":
            return "No file was selected"
        
        if file and file_is_allowed(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # This should not be done in real servers
            # but for the sake of simplicity
            file.save(filepath)
            
            image = cv2.imread(filepath)
            resized_img = cv2.resize(image, (224, 224))
            edges = cv2.Canny(resized_img, threshold1=70, threshold2=70)
            
            image_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # let me overwrite the already upoaded image
            edges_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filename=edges_img_path, img=edges)
            # end test
            
            image_batch = np.expand_dims(image_rgb, axis=0)
            
            prediction = arabic_model.predict(image_batch)
            predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability
            
            
            predicted_class_one_hot = np.zeros((predicted_class_index.size, arabic_onehot_encoder.categories_[0].size))
            predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
            predicted_class_label = arabic_onehot_encoder.inverse_transform(predicted_class_one_hot)[0][0]
            
            
            return render_template("arabic_classification.html", show_result = True, predicted_class=predicted_class_label, edges_image=filename)
        

@app.route("/english", methods=['GET', 'POST'])
def english_classification():
    if request.method == "GET":
        return render_template("english_classification.html", show_result=False)
    else:
        
        if 'file' not in request.files:
            return "No file was uploaded"
        
        file = request.files['file']
        
        if file.filename == "":
            return "No file was selected"
        
        if file and file_is_allowed(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # This should not be done in real servers
            # but for the sake of simplicity
            file.save(filepath)
            
            image = cv2.imread(filepath)
            resized_img = cv2.resize(image, (224, 224))
            edges = cv2.Canny(resized_img, threshold1=70, threshold2=70)
            
            image_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # let me overwrite the already upoaded image
            edges_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filename=edges_img_path, img=edges)
            # end test
            
            image_batch = np.expand_dims(image_rgb, axis=0)
            
            prediction = english_model.predict(image_batch)
            predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability
            
            
            predicted_class_one_hot = np.zeros((predicted_class_index.size, english_onehot_encoder.categories_[0].size))
            predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
            predicted_class_label = english_onehot_encoder.inverse_transform(predicted_class_one_hot)[0][0]
            
            
            return render_template("english_classification.html", show_result = True, predicted_class=predicted_class_label, edges_image=filename)
        


if __name__ == "__main__":
    app.run(debug=True, port=4000)