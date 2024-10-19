import os
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import numpy as np
from tensorflow.keras.models import load_model #type: ignore
from transformers import MarianMTModel, MarianTokenizer
from consts import single_char_translations


arabic_model = load_model('./models/arabic_model.h5')
arabic_encoder = joblib.load("./models/arabic_encoder.pkl")


english_model = load_model('./models/english_model.h5')
english_encoder = joblib.load("./models/english_encoder.pkl")


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def file_is_allowed(filename: str):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def get_image_prediction(file, model, encoder) -> str:
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
    
    prediction = model.predict(image_batch)
    predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability
    
    
    predicted_class_one_hot = np.zeros((predicted_class_index.size, encoder.categories_[0].size))
    predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
    predicted_class_label = encoder.inverse_transform(predicted_class_one_hot)[0][0]
    
    return filename, predicted_class_label            

def translate_text(input_text, source_lang, target_lang):
    # Check if input_text is a single character and translate manually
    if source_lang=='en':
        if len(input_text) == 1 and input_text.lower() in single_char_translations:
            return single_char_translations[input_text.lower()]

    # For words and sentences, use MarianMT for translation
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    
    # Load the pre-trained model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Ensure input_text is a string or flatten it if it's an array
    if isinstance(input_text, np.ndarray):
        input_text = input_text.flatten().astype(str).tolist()  # Convert to list of strings
        input_text = ' '.join(input_text)  # Join list into a single string
    elif isinstance(input_text, list):
        input_text = ' '.join(input_text)
    elif not isinstance(input_text, str):
        raise ValueError("Input text must be a string, a list of strings, or a numpy.ndarray.")
    
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Translate text with max_new_tokens
    output_ids = model.generate(input_ids, max_new_tokens=50)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return translated_text.lower()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)            

                
@app.route("/img/<language>", methods=['GET', 'POST'])
def classification(language):
    
    if language not in ['arabic', 'english']:
        return "Invalid language", 404
    
    if request.method == "GET":
        return render_template(f"classification.html", show_result=False, language=language)
    else:
        
        if 'file' not in request.files:
            return "No file was uploaded"
        
        file = request.files['file']
        
        if file.filename == "":
            return "No file was selected"
        
        if file and file_is_allowed(file.filename):
            
            model = arabic_model if language == "arabic" else english_model
            encoder = arabic_encoder if language == "arabic" else english_encoder
            
            name, predicted_class_label = get_image_prediction(file=file, model=model, encoder=encoder)
            
            return render_template(f"classification.html", show_result = True, predicted_class=predicted_class_label, edges_image=name, language=language)
        

if __name__ == "__main__":
    app.run(debug=True, port=4000)