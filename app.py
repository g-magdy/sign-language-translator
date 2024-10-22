'''
1. mapping characters
2. word translation
3. sentence translation

4. 
'''



import os
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory
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
FileItem = namedtuple('FileItem', ['filename', 'file'])
# import spacy

# arabic_model_path = 'C:/Users/gsags/Downloads/arabic_model.h5'
# english_model_path = '"C:/Users/gsags/Downloads/english_model.h5'

arabic_model = load_model('C:/Users/gsags/Downloads/arabic_model.h5')
english_model = load_model('C:/Users/gsags/Downloads/english_model.h5')

# if os.path.exists(arabic_model):
#     print(f"Found Arabic model at: {arabic_model}, size: {os.path.getsize(arabic_model)} bytes")
# else:
#     raise (f"Arabic model file not found: {arabic_model}")

# if os.path.exists(english_model):
#     print(f"Found English model at: {english_model}, size: {os.path.getsize(arabic_model)} bytes")
# else:
#     raise (f"English model file not found: {english_model}")


arabic_encoder = joblib.load("./models/arabic_encoder.pkl")
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


def get_image_predictions_multi(image_paths, model, encoder) -> tuple:
    filenames = []  # To store filenames
    predicted_labels = []  # To store predicted class labels

    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip if the image is not found

        # Resize and process the image
        resized_img = cv2.resize(image, (224, 224))
        edges = cv2.Canny(resized_img, threshold1=70, threshold2=70)
        image_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Prepare the image batch for prediction
        image_batch = np.expand_dims(image_rgb, axis=0)

        # Make prediction
        prediction = model.predict(image_batch)
        predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability

        # Convert the predicted index to one-hot and then to label
        predicted_class_one_hot = np.zeros((predicted_class_index.size, encoder.categories_[0].size))
        predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
        predicted_class_label = encoder.inverse_transform(predicted_class_one_hot)[0][0]

        # Append filename and predicted class label to their respective lists
        filenames.append(os.path.basename(image_path))
        predicted_labels.append(predicted_class_label)

    return filenames, predicted_labels

          

def translate_text(input_text, source_lang, target_lang):
    # Check if input_text is a single character and translate manually
    if source_lang=='en':
        if len(input_text) == 1 and input_text.lower() in consts.en_ar_map:
            return consts.en_ar_map[input_text.lower()]

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


# Load the NLP model (make sure to have 'en_core_web_sm' downloaded)
# nlp = spacy.load('en_core_web_sm')
# Download nltk corpus data
nltk.download('punkt')

# Define global variables to accumulate letters and words
accumulated_letters_asl = []
accumulated_words_asl = []

def classify_and_accumulate_letters_asl(asl_image_paths, asl_model, delimiter=None):
    """Classify a list of images and accumulate the predicted letters, handling word boundaries."""
    global accumulated_letters_asl

    for asl_image_path in asl_image_paths:
        if delimiter is not None:
            finalize_and_accumulate_word_asl()  # Finalize the current word if a delimiter is provided
        else:
            predicted_asl_letter = predict_image_asl(asl_model, asl_image_path)
            predicted_asl_letter = predicted_asl_letter.item() if isinstance(predicted_asl_letter, np.ndarray) else predicted_asl_letter

            # Check if the predicted letter is valid before appending
            if predicted_asl_letter:
                accumulated_letters_asl.append(predicted_asl_letter)
                print(f'Accumulated letters so far: {accumulated_letters_asl}')
            else:
                print(f'No valid prediction for image: {asl_image_path}')


# Global variable to hold accumulated words for the sentence
accumulated_words_asl = []

def finalize_and_accumulate_word_asl():
    """Finalizes the current accumulation of letters and handles word boundaries."""
    global accumulated_letters_asl, accumulated_words_asl
    
    if accumulated_letters_asl:
        # Combine letters into a single string without spaces
        combined_word = ''.join(accumulated_letters_asl)

        # Add the finalized word to the accumulated words list
        accumulated_words_asl.append(combined_word)
        print(f'Finalized word: {combined_word}')

        # Clear accumulated letters for the next word
        accumulated_letters_asl.clear()
    else:
        print('No letters accumulated for the current word.')

def construct_final_sentence():
    """Construct the final sentence from accumulated words."""
    if accumulated_words_asl:
        # Join words with spaces and capitalize the first letter
        final_sentence = ' '.join(accumulated_words_asl).capitalize() + '.'
        print(f'Final sentence to translate: "{final_sentence}"')
    else:
        print('No words accumulated to form a sentence.')

def translate_text(sentence, source_lang='en', target_lang='ar'):
    """Translate a given sentence from the source language to the target language."""
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize and translate
    tokenized_text = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    translated_tokens = model.generate(**tokenized_text)
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_sentence

def finalize_and_translate_sentence_asl():
    global accumulated_words_asl

    # Join and prepare the final sentence
    if accumulated_words_asl:
        sentence = ' '.join(accumulated_words_asl).capitalize() + '.'
    else:
        sentence = ''

    print(f'Final sentence to translate: "{sentence}"')  # Debug output

    # Remove the period for translation purposes
    sentence_to_translate = sentence.rstrip('.')

    # Translate the constructed sentence to Arabic without the period
    translated_sentence_asl = translate_text(sentence_to_translate, source_lang='en', target_lang='ar')
    
    # Remove any unwanted parentheses in the translation
    translated_sentence_asl = translated_sentence_asl.replace('(', '').replace(')', '')
    
    print(f'Translated sentence: "{translated_sentence_asl}"')  # Debug output

    # Re-attach the period after translation
    translated_sentence_asl += '.'

    # Custom function to split Arabic words into characters
    def split_arabic_words_to_letters(arabic_sentence):
        letters = []
        for word in arabic_sentence.split():
            i = 0
            while i < len(word):
                # Check for "ال"
                if word[i:i+2] == 'ال':
                    letters.append('ال')  # Add 'ال' as a single character
                    i += 2  # Move index forward by 2
                else:
                    letters.append(word[i])  # Add the current character
                    i += 1  # Move to the next character
        return letters

    # Split the translated Arabic sentence for processing
    split_letters = split_arabic_words_to_letters(translated_sentence_asl)
    
    for letter in split_letters:
        display_arsl_image(letter)  # Display corresponding Arabic sign language image
    
    # Clear accumulated words after translation
    accumulated_words_asl = []

def split_into_words(letter_sequence):
    """Use spaCy to predict word boundaries in the given letter sequence."""
    doc = nlp(letter_sequence)
    words = [token.text for token in doc]
    return words

def classify_and_translate_images(arsl_image_paths, arsl_model):
    # Step 1: Predict the ArSL letters from the input images
    predicted_arsl_letters = predict_image_multi(arsl_model, arsl_image_paths)
    # print(f'Predicted ArSL letters: {predicted_arsl_letters}')
    
    # Step 2: Extract the actual letter from the numpy arrays and join to form the word
    arsl_word = ''.join([letter.item() if isinstance(letter, np.ndarray) else letter for letter in predicted_arsl_letters])
    # print(f'Formed ArSL word: {arsl_word}')
    
    # Step 3: Translate the ArSL word to English
    translated_word = translate_text_multi(arsl_word, source_lang='ar', target_lang='en')
    if translated_word:
        english_word = str(translated_word[0]).strip(string.punctuation)  # assuming translate_text returns a list
        # print(f'Translated to English: {english_word}')
        
        # Step 4: Split the English word into individual letters
        english_letters = list(english_word)
        # print(f'Split English word into letters: {english_letters}')
        
        # Step 5: Display the ASL images corresponding to each English letter
        display_asl_image_multi(english_letters)
    # else:
        # print('Translation failed or no corresponding word found.')


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
        

@app.route("/word/<language>", methods=['GET', 'POST'])
def translate_word(language):
    
    if language not in ['arabic', 'english']:
        return "Invalid language", 404

    
    if request.method == "GET":
        return render_template(f"word_translation.html", show_result=False, language=language)
    else:

        model = arabic_model if language == "arabic" else english_model
        encoder = arabic_encoder if language == "arabic" else english_encoder
        files = request.files.getlist('images[]')
        if files and len(files) > 0:
            # Save files temporarily and get their paths
            image_paths = []
            for file in files:
                # Define a temporary path (you can customize this as needed)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(temp_path)  # Save the uploaded file
                image_paths.append(temp_path)
    # Now you can process the images in the order they were uploaded
        filenames, predicted_labels = get_image_predictions_multi(image_paths=image_paths, model=model, encoder=encoder)
        return render_template(f"classification.html", show_result = True, predicted_class=predicted_labels, edges_image=filenames, language=language)
        
        # for file in files:
        #     if file.filename != "":
        #         return 'Files received successfully'
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))


        
        # return "saved"




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)