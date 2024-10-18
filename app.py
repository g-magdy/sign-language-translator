from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/upload", methods=['GET', 'POST'])
def upload_photo():
    if request.method == "GET":
        return render_template("upload.html")
    else:
        return "TODO: post part"

if __name__ == "__main__":
    app.run(debug=True, port=4000)