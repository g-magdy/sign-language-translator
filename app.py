'''
1. mapping characters
2. word translation
3. sentence translation

4. 
'''



import os
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory,jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.tokenize import word_tokenize
import string
from collections import namedtuple
import consts


arabic_model = load_model('C:/Users/gsags/Downloads/arabic_model.h5')
english_model = load_model('C:/Users/gsags/Downloads/english_model.h5')


arabic_encoder = joblib.load("./models/arabic_encoder.pkl")
english_encoder = joblib.load("./models/english_encoder.pkl")

# Dictionary mapping languages to their respective alphabets
alphabets = {
    'english': list('abcdefghijklmnopqrstuvwxyz'),
    'arabic': ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    # Add more languages and their alphabets here
}


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def file_is_allowed(filename: str):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def get_image_prediction(file, model, encoder) -> str:
    # Load the image
    nparr = np.frombuffer(file.read(), np.uint8)  # Use frombuffer instead of fromstring
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Could not open or find the image.")
    
    # Resize image to (224, 224)
    img_resized = cv2.resize(image, (224, 224))

    # Step 1: Grayscale conversion
    gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # You can adjust the thresholds

    # Step 3: Prepare the image for prediction
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges back to 3 channels (RGB)
    image_batch = np.expand_dims(edges_rgb, axis=0)  # Add batch dimension (shape: (1, 224, 224, 3))

    # Perform prediction
    prediction = model.predict(image_batch)
    predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability

    # One-hot encoding for inverse transformation
    predicted_class_one_hot = np.zeros((predicted_class_index.size, encoder.categories_[0].size))
    predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
    predicted_class_label = encoder.inverse_transform(predicted_class_one_hot)[0][0]
    # Convert predicted_class_label to string if it's not already
    if isinstance(predicted_class_label, np.ndarray):
        predicted_class_label = predicted_class_label.item()  # Convert to a single string if it's an array

    return str(predicted_class_label)  # Ensure return types are str and bool



@app.route('/')
def index():
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)            

def get_model_for_language(language):
    # Replace with logic to return the appropriate model based on the language
    if language == "arabic":
        return arabic_model  # Your Arabic model
    elif language == "english":
        return english_model  # Your English model
    # Add more languages as necessary

def get_encoder_for_language(language):
    # Replace with logic to return the appropriate encoder based on the language
    if language == "arabic":
        return arabic_encoder  # Your Arabic encoder
    elif language == "english":
        return english_encoder  # Your English encoder
    # Add more languages as necessary


                
@app.route("/txt/<language>", methods=['GET', 'POST'])
def classification(language):
    # Get alphabet based on the language
    alphabet = alphabets.get(language, alphabets['english'])
    if language not in ['arabic', 'english']:
        return "Invalid language", 404

    if request.method == "POST":
        # Handle the selected letter from the form
        selected_letter = request.form.get("alphabet")
        if selected_letter:
            jpg_path = f"/static/{language}/{selected_letter}.jpg"
            jpeg_path = f"/static/{language}/{selected_letter}.jpeg"
            image_path = None
            
            if os.path.exists(f".{jpg_path}"):  # Check for .jpg file
                image_path = jpg_path
            elif os.path.exists(f".{jpeg_path}"):  # Check for .jpeg file
                image_path = jpeg_path

            # Return JSON response with the image path and selected letter
            return jsonify({"image_path": image_path, "selected_letter": selected_letter})

        # Handle image file upload
        uploaded_file = request.files.get("image")  # Assuming the input name is "image"
        if uploaded_file:
            model = get_model_for_language(language)  # Function to retrieve the model for the selected language
            encoder = get_encoder_for_language(language)  # Function to retrieve the encoder for the selected language

            # Call the get_image_prediction function
            predicted_label= get_image_prediction(uploaded_file, model, encoder)

            # Return JSON response with prediction result
            return jsonify({"predicted_label": predicted_label})

    # Render the full template for GET requests
    return render_template(
        "classification.html",
        show_result=False,
        language=language,
        alphabet=alphabet,
        image_path=None
    )



@app.route("/name/<language>", methods=['GET', 'POST'])
def handle_name(language):
    # Print the method type
    # print(f"Request method: {request.method}")

    # Check if the request is POST
    if request.method == 'POST':
        # print(f"Request files: {request.files}")  # Debugging line
        # print(f"Request form: {request.form}")    # Debugging line

        # Check if we're handling name or image upload
        if 'name' in request.form:  # Handle name input
            name = request.form.get("name", "").lower()  # Get name and convert to lowercase
            # Validate name input
            if not name.isalpha() or len(name) < 2:  # Check if name is valid
                return jsonify({"error": "Invalid name"}), 400

            characters = list(name)  # Get individual characters
            images = []

            # Collect image paths for each character
            for char in characters:
                char_image_path_jpg = f"/static/{language}/{char}.jpg"
                char_image_path_jpeg = f"/static/{language}/{char}.jpeg"

                # Check for the existence of image files
                if os.path.exists(f".{char_image_path_jpg}"):
                    images.append(char_image_path_jpg)
                elif os.path.exists(f".{char_image_path_jpeg}"):
                    images.append(char_image_path_jpeg)
                else:
                    return jsonify({"error": f"No image found for character '{char}'"}), 404

            if not images:
                return jsonify({"error": "No images found"}), 404

            return jsonify({"images": images, "letters": characters})

        # Handle image upload for prediction
        uploaded_file = request.files.get("image")
        if uploaded_file:
            # print("Uploaded file name:", uploaded_file.filename)  # Debugging line
            model = get_model_for_language(language)
            encoder = get_encoder_for_language(language)

            predicted_label = get_image_prediction(uploaded_file, model, encoder)

            return jsonify({"predicted_label": predicted_label})

        return jsonify({"error": "No valid data provided"}), 400

    return jsonify({"error": "Invalid request method"}), 405



@app.route('/cross_language')
def cross_language():
    return render_template('cross_language.html')

@app.route('/sign_to_sign')
def sign_to_sign():
    return render_template('sign_to_sign.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)