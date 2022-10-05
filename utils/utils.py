
import tensorflow as tf
import numpy as np
from utils.data_utils import max_length, num_to_char
import matplotlib.pyplot as plt


def decode_prediction(prediction: np.array) -> list:
    '''
    :param prediction: prediction of model
    :return decoded output of model
    '''
    input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
    # decode prediction by ctc decoder
    results = tf.keras.backend.ctc_decode(
        prediction, input_length=input_length, greedy=True)[0][0][:, :max_length]

    output = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output.append(res)

    return output


# a function for plotting a desired image and its prediction
def plot_prediction(image_path: str, prediction: str) -> None:
    '''
    :param image_path: path of image 
    :param prediction: predicted label of image
    '''

    plt.figure()
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    plt.show()


# utility function for plotting loss
def plot_loss(history: tf.keras.callbacks.History) -> None:
    '''
    ploting the losses of history object
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()

    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('#epochs')
    plt.ylabel('loss')

    plt.show()
