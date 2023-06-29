from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as  np
import tensorflow as tf

app = Flask(__name__)

dic = {0 : 'Áo thun', 1 : 'Quần dài', 2: 'Áo len', 3: 'Đầm', 4: 'Áo khoác',
        5: 'Sandal', 6: 'Áo sơ mi', 7:'Giày thể thao', 8: 'Túi xách', 9: 'Ủng'}

model = load_model('VGG16_50epoch.h5')
model.make_predict_function()

def predict_label(path_file):
	image = cv2.imread(path_file)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image,(28,28))

	image = 1 - image/255
	image = np.expand_dims(image, 2)
	image = tf.expand_dims(image, 0)

	pre = model.predict(image)
	prediction=np.argmax(pre,axis=1)
	print([prediction[0]])
	return dic[prediction[0]]
    
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)
	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)