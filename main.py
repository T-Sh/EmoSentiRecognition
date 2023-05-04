from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from util.predict.predictor import Predictor


app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/video': {"origins": "http://localhost:8080"}})

model_name = 'Tatyana/dusha_intermodal_4_emotions'
predictor = Predictor(model_name)


@app.route('/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Video successfully uploaded')
        label = predictor.predict(filename)
        return label


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
