from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('model/digit_generator_model.h5')

def generate_images(digit, n=5):
    images = []
    while len(images) < n:
        idx = np.random.choice(np.where(np.argmax(model.predict(x_train[:1000]), axis=1) == digit)[0])
        img = x_train[idx].reshape(28, 28)
        images.append(img)
    return images

@app.route("/", methods=["GET", "POST"])
def index():
    images = []
    digit = None
    if request.method == "POST":
        digit = int(request.form["digit"])
        imgs = generate_images(digit)
        os.makedirs('static/generated', exist_ok=True)
        images = []
        for i, img_array in enumerate(imgs):
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            path = f"static/generated/digit_{i}.png"
            img.save(path)
            images.append(path)
    return render_template("index.html", images=images, digit=digit)

if __name__ == "__main__":
    app.run(debug=True)
