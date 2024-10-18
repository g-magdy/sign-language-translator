import os
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

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
            
            image = Image.open(filepath)
            greyscale_img = image.convert('L')
            greyscale_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'greyscale_' + filename)
            greyscale_img.save(greyscale_img_path)
            
            return render_template("upload.html", filename='greyscale_' + filename)
            

if __name__ == "__main__":
    app.run(debug=True, port=4000)