import tensorflow as tf
from tensorflow import keras
from model import make_model
from utils.data_utils import train_test_split, read_image, gathering_data
import numpy as np
from utils.utils import plot_loss
import sys


# set default batch size and number of epochs
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
epochs = 50


# set first command line argument as batch_size and second one as number of epochs

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
if len(sys.argv) > 2:
    epochs = int(sys.argv[2])

# gathering dataset

data_path = './samples/'
images, labels = gathering_data(data_path)

# spliting train and test sets
x_train, y_train, x_test, y_test = train_test_split(
    np.array(images), np.array(labels))

# use tf.data for batching and pipelining
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

ds_train = ds_train.map(read_image, num_parallel_calls=AUTOTUNE).batch(
    batch_size).prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_val = ds_val.map(read_image, num_parallel_calls=AUTOTUNE).batch(
    batch_size).prefetch(AUTOTUNE)


# create model and compile it
model = make_model()

model.compile(
    optimizer=keras.optimizers.Adam(),
)

# model.summary()

# stop training if after 5 epochs our model won't imporove its loss
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# logging training and val losses
csv_logger = keras.callbacks.CSVLogger('training.csv')

# keras.utils.plot_model(model,show_shapes=True)

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=epochs,
    callbacks=[early_stopping, csv_logger]
)

# plotting training and validation losses
plot_loss(history)


# we save model for later usages....
# if you want to save model uncomment below lines then run the code

# print('saving model....')
# model.save('ocr_model/')
# print('model saved!!')
