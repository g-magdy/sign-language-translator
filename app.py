import os
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import numpy as np
from tensorflow.keras.models import load_model

vgg19_model = load_model('./models/testvgg_19.h5')
arsl_model = load_model('./models/arsl_model_2.h5')
arsl_model = load_model('./models/asl_model_2.h5')
arabic_onehot_encoder = joblib.load("./models/arabic_onehot_encoder.pkl")

def get_label(code):
    labels = {
        0: 'buildings',
        1: 'forest',
        2: 'glacier',
        3: 'mountain',
        4: 'sea',
        5: 'street'
    }
    
    return labels[code]


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


@app.route("/upload", methods=['GET', 'POST'])
def upload_photo():
    if request.method == "GET":
        return render_template("upload.html")
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
            image_resized = cv2.resize(image, (224, 224))
            gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, threshold1=70, threshold2=70)
            
            edges_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'e_' + filename)
            cv2.imwrite(filename=edges_img_path, img=edges)
            
            return render_template("upload.html", filename='e_' + filename)
            
            
@app.route("/classify", methods=['GET', 'POST'])
def classify_img():
    if request.method == "GET":
        return render_template("classify.html", show_result=False)
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
            
            image = Image.open(filepath)
            
            # TODO: Apply processing here
            image = image.resize((150, 150))
            image_array = np.array(image) 
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
             # Make a prediction
            prediction = vgg19_model.predict(image_array)
        
            # Example: Get the predicted class (assuming the model outputs a single value)
            predicted_class_code = np.argmax(prediction)
            
            predicted_class = get_label(predicted_class_code)
            
            return render_template("classify.html", show_result=True, predicted_class=predicted_class)
    
@app.route("/arsl", methods=['GET', 'POST'])
def arsl_classification():
    if request.method == "GET":
        return render_template("arsl_classification.html", show_result=False)
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
            
            image_batch = np.expand_dims(image_rgb, axis=0)
            
            prediction = arsl_model.predict(image_batch)
            predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability
            
            predicted_class_one_hot = np.zeros((predicted_class_index.size, arabic_onehot_encoder.categories_[0].size))
            predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
            predicted_class_label = arabic_onehot_encoder.inverse_transform(predicted_class_one_hot)
            
            print(arabic_onehot_encoder.categories_)
            
            return render_template("arsl_classification.html", show_result = True, predicted_class=predicted_class_label)

if __name__ == "__main__":
    app.run(debug=True, port=4000)