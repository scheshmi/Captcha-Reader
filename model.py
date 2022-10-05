
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from utils.data_utils import image_height, image_width, char_to_num


class CTC(keras.layers.Layer):
    '''
    implement ctc loss layer
    '''

    def __init__(self) -> None:
        super(CTC, self).__init__()
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_length = tf.cast(tf.shape(y_true)[0], dtype=tf.int64)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)
        label_length = tf.cast(tf.shape(y_true)[1], dtype=tf.int64)

        input_length = input_length * \
            tf.ones(shape=(batch_length, 1), dtype=tf.int64)
        label_length = label_length * \
            tf.ones(shape=(batch_length, 1), dtype=tf.int64)

        # compute cost and add it to layer
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


def make_model() -> keras.models.Model:
    """
    Architecture
    Image input -> Conv1 ->BATCHNORM -> RELU -> MAXPOOL2D -> 
    CONV2 ->BATCHNORM -> RELU -> MAXPOOL2D ->
    RESHAPE -> DENSE -> RELU -> DROPOUT ->
    LSTM1 -> LSTM2 -> DENSE -> SOFTMAX ->
    CTC LOSS

    :return defined model 
    """
    # defining model inputs
    input = keras.layers.Input(
        shape=(image_width, image_height, 1),
        name='image',
        dtype=tf.float32
    )
    labels = keras.layers.Input(
        shape=(None,),
        name='label',
        dtype=tf.float32
    )

    # conv1
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
    )(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # conv2
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # reshaping
    # we downsized the image 4x and as we have 64 filters in last conv reshape as this way to pass it to RNNs
    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape,)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # RNN

    x = layers.Bidirectional(layers.LSTM(
        128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(
        64, return_sequences=True, dropout=0.25))(x)

    # output layer
    x = layers.Dense(len(char_to_num.get_vocabulary()) +
                     1, activation="softmax")(x)

    # calculating CTC loss

    output = CTC()(labels, x)

    # creating model
    model = keras.models.Model(
        inputs=[input, labels],
        outputs=output
    )

    return model
