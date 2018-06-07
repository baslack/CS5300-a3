import tensorflow as tf

import os
import numpy as np


def import_mnist(filepath):
    import gzip
    image_files = [
        "t10k-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz"
    ]
    label_files = [
        "t10k-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz"
    ]
    image_data = dict()
    for this_file in image_files + label_files:
        with gzip.open(os.path.join(filepath, this_file), "rb") as f:
            f.seek(2)
            magic = f.read(1)
            size_dict = {
                b'\x08': np.uint8,
                b'\x09': np.int8,
                b'\x0B': np.int16,
                b'\x0C': np.int32,
                b'\x0D': np.float32,
                b'\x0E': np.float64
            }
            dim = int.from_bytes(f.read(1), byteorder="big")
            shape = list()
            for this_dim in range(dim):
                shape.append(int.from_bytes(f.read(4), byteorder="big"))
            raw_bytes = f.read(-1)
            ar = np.frombuffer(raw_bytes, dtype=size_dict[magic]).reshape(shape)
            image_data[this_file] = {
                "magic": magic,
                "dim": dim,
                "shape": shape,
                "ar": ar,
                "dtype": size_dict[magic]
            }

    return image_data["train-images-idx3-ubyte.gz"], \
           image_data["train-labels-idx1-ubyte.gz"], \
           image_data["t10k-images-idx3-ubyte.gz"], \
           image_data["t10k-labels-idx1-ubyte.gz"]


def mnist_train_input_fn(to_features, to_labels, batch_size=128):
    features = tf.data.Dataset.from_tensor_slices(to_features).map(lambda x: x / 255)
    one_hot = tf.one_hot(list(range(10)), 10, on_value=1, off_value=0)
    labels = tf.cast(to_labels, dtype=tf.int32)
    labels = tf.data.Dataset.from_tensor_slices(labels).map(lambda x: one_hot[x, :])

    temp_dataset = tf.data.Dataset.zip((features, labels))
    temp_dataset = temp_dataset.shuffle(to_features.shape[0]).repeat().batch(batch_size)
    return temp_dataset


def mnist_eval_input_fn(to_features, to_labels):
    features = tf.data.Dataset.from_tensor_slices(to_features).map(lambda x: x / 255)
    one_hot = tf.one_hot(list(range(10)), 10, on_value=1, off_value=0)
    labels = tf.cast(to_labels, dtype=tf.int32)
    labels = tf.data.Dataset.from_tensor_slices(labels).map(lambda x: one_hot[x, :])

    temp_dataset = tf.data.Dataset.zip((features, labels))
    # temp_dataset = temp_dataset.shuffle(to_features.shape[0]).repeat().batch(batch_size)
    return temp_dataset


MNIST_DATA_DIR = r"C:\Dropbox\CS5300\tensorflow_mnist\MNIST-data"

if __name__ == "__main__":
    from a3.cnn import cnn_model_fn
    from a3.nn import nn_model_fn

    train_x, train_y, test_x, test_y = import_mnist(MNIST_DATA_DIR)
    temp_dir = os.path.expanduser("~/Desktop/mnist_cnn_test")
    params = {
        "activation": tf.nn.leaky_relu,
        "use_bias": True,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "logdir": temp_dir,
        "features_shape": (-1, 28, 28, 1),
        "labels_shape": (-1, 10)
    }
    cnn = tf.estimator.Estimator(cnn_model_fn,
                                 model_dir=temp_dir,
                                 params=params)

    for i in range(6):
        cnn.train(lambda: mnist_train_input_fn(train_x["ar"], train_y["ar"]), steps=1000)
        cnn.evaluate(lambda: mnist_eval_input_fn(test_x["ar"], test_y["ar"]))
    preds = cnn.predict(lambda: mnist_eval_input_fn(test_x["ar"], test_y["ar"]), predict_keys='class_ids')
    for this in preds:
        print(this['class_ids'][0])
        break

    temp_dir = os.path.expanduser("~/Desktop/mnist_nn_test")
    params["features_shape"] = (-1, 784)
    params["units"] = [2048, 2048, 2048]
    nn = tf.estimator.Estimator(nn_model_fn,
                                model_dir=temp_dir,
                                params=params)

    for i in range(6):
        nn.train(lambda: mnist_train_input_fn(train_x["ar"], train_y["ar"]), steps=1000)
        nn.evaluate(lambda: mnist_eval_input_fn(test_x["ar"], test_y["ar"]))
    preds = nn.predict(lambda: mnist_eval_input_fn(test_x["ar"], test_y["ar"]), predict_keys='class_ids')
    for this in preds:
        print(this['class_ids'][0])
        break
