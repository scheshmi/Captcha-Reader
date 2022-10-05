
import tensorflow as tf
from utils.utils import decode_prediction, plot_prediction
from utils.data_utils import read_image
import sys

# load trained model
model = tf.keras.models.load_model('./ocr_model/')

# model.summary()

# get output layer of last dense layer and inputs of first layer
prediction_model = tf.keras.models.Model(
    model.get_layer(name='image').input,
    model.get_layer(name='dense_1').output
)


# prediction_model.summary()


########## prediction on a random sample ############
# example ./samples/2enf4.png

# geting first argument as path of image which want to predict

image_path = sys.argv[1]

img_label = read_image(image_path, '')
img = tf.expand_dims(img_label['image'], 0)

pred = prediction_model.predict(img)

decoded_pred = decode_prediction(pred)

plot_prediction(image_path, decoded_pred)
