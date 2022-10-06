from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from flask_bootstrap import Bootstrap
from PIL import Image
import numpy as np
import os

classes = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"]
num_classes = len(classes)
image_size = 224

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bootstrap = Bootstrap(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.name == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model('./ResNet.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image) / 255
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return render_template('predict.html', name=classes[predicted])
            # "ラベル： " + classes[predicted] + ", 確率： " + str(percentage) + "%"

    return render_template('upload.html')
