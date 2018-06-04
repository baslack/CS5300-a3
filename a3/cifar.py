import tensorflow as tf
import numpy as np


def import_cifar(import_dir):
    import pickle
    import os

    training_files = ["data_batch_{}".format(x + 1) for x in range(5)]
    training_files = [os.path.join(import_dir, x) for x in training_files]
    test_file = os.path.join(import_dir, "test_batch")
    labels_file = os.path.join(import_dir, "batches.meta")
    train_data = np.empty(shape=(0, 3072), dtype="uint8")
    train_labels = list()
    for this_file in training_files:
        with open(this_file, "rb") as f:
            temp_dict = pickle.load(f, encoding="bytes")
            train_data = np.concatenate((train_data, temp_dict[b"data"]), axis=0)
            train_labels.extend(temp_dict[b"labels"])
    with open(test_file, "rb") as f:
        temp_dict = pickle.load(f, encoding="bytes")
        test_data = temp_dict[b"data"]
        test_labels = temp_dict[b"labels"]
    with open(labels_file, "rb") as f:
        label_names = pickle.load(f, encoding="bytes")[b"label_names"]
    return train_data, train_labels, test_data, test_labels, label_names


def prep_dataset(x, y, bReshape=True):
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     {"x": tf.placeholder(tf.int8, shape=(1, 3072,)),
    #      "y": tf.placeholder(tf.int32, shape=(1, 1))}
    # )
    dataset = tf.data.Dataset.from_tensor_slices({
        "x": x,
        "y": y
    })
    dataset = dataset.map(lambda d: {"x": d["x"] / 255, "y": d["y"]})
    if bReshape:
        dataset = dataset.map(lambda d: {"x": tf.reshape(d["x"], (-1, 32, 32, 3)), "y": d["y"]})
    one_hot = tf.one_hot(list(range(10)), 10, on_value=1, off_value=0)
    # print(one_hot)
    dataset = dataset.map(lambda d: {"x": d["x"], "y": one_hot[d["y"], :]})
    features = dataset.map(lambda d: d["x"])
    labels = dataset.map(lambda d: d["y"])
    return features, labels


data_path = "C:\\Dropbox\\CS5300\\tensorflow_cifar_scratch\\CIFAR_data\\cifar-10-batches-py"

if __name__ == "__main__":
    train_x, train_y, test_x, test_y, labels = import_cifar(data_path)
    features, labels = prep_dataset(train_x, train_y)
    temp_dataset = tf.data.Dataset.zip((features, labels)).shuffle(50000).repeat().batch(5)
    next_item = temp_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for i in range(10):
            print(sess.run(next_item))
