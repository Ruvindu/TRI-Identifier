from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64,operator
from io import BytesIO


from flask import Flask,request,render_template,jsonify

app = Flask(__name__)
# Load the model
model = load_model('keras_model.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictbycapture', methods=['POST'])
def predictbycapture():

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # image = Image.open(r'img_20220605_123607.jpg')
    # convert captured 64based img to image
    image = Image.open( BytesIO(base64.b64decode(request.form['test_img'].split(',')[1])) )

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # Lables
    # 0 2023
    # 1 2026
    # 2 2025

    tri2023 = round(prediction[0][0] * 100,2)
    tri2026 = round(prediction[0][1] * 100,2)
    tri2025 = round(prediction[0][2] * 100,2)

    if (tri2023 > tri2026) and (tri2023 > tri2025):
        max_val = tri2023
        max_clone = "TRI 2023"
    elif (tri2026 > tri2023) and (tri2026 > tri2025):
        max_val = tri2026
        max_clone = "TRI 2026"
    else:
        max_val = tri2025
        max_clone = "TRI 2025"

    return jsonify(tri2023=tri2023,tri2026=tri2026,tri2025=tri2025,max_val=max_val,max_clone=max_clone)


if __name__ == '__main__':
    app.run()
