
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from pathlib import Path
import os


def gathering_data(data_dir: str = './samples/') -> tuple:
    '''
    :param data_dir : directory of dataset
    :return images and labels of them
    '''
    data_dir = Path(data_dir)

    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]

    return images, labels


data_path = './samples/'
images, labels = gathering_data(data_path)

# characters are represented in labels
characters = sorted(list(set(char for label in labels for char in label)))

max_length = max([len(label) for label in labels])

# mapping from chars to numbers
char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

# mapping from numbers to chars
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)


def train_test_split(images: np.array, labels: np.array, train_size: float = 0.8, shuffle: bool = True) -> tuple:
    '''
    :param images: images 
    :param labels: labels of images
    :param train_size: portion of dataset we want to be as train set
    :param shuffle: set it to true if you want to shuffle dataset
    :return train and test datasets and their labels
    '''

    dataset_size = len(images)

    train_index = int(train_size * dataset_size)

    indices = np.arange(dataset_size)

    if shuffle:
        np.random.shuffle(indices)

    x_train, y_train = images[indices[:train_index]
                              ], labels[indices[:train_index]]
    x_test, y_test = images[indices[train_index:]
                            ], labels[indices[train_index:]]

    return x_train, y_train, x_test, y_test


image_height = 50
image_width = 200


def read_image(file_path: str, label: str) -> tuple:
    '''
    :param file_path: path of file which want to read
    :param label: label of image
    '''

    img = tf.io.read_file(filename=file_path)

    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, [image_height, image_width])
    # transpose image because we want to time dimension corresponds to width
    img = tf.transpose(img, perm=[1, 0, 2])

    label = char_to_num(tf.strings.unicode_split(
        label, input_encoding="UTF-8"))
    # because our model gets 2 inputs we return as this way
    return {"image": img, "label": label}
