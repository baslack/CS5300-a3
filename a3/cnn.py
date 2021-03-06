import tensorflow as tf
from typing import Dict
from a3.cifar import import_cifar, data_path
import os


def cnn_model_fn(features: tf.Tensor,
                 labels: tf.Tensor,
                 mode: tf.estimator.ModeKeys,
                 params: Dict) -> tf.estimator.EstimatorSpec:
    weight_decay = 1e-4

    input_ = tf.reshape(features, params["features_shape"])

    if labels is not None:
        labels = tf.reshape(labels, params["labels_shape"])

    # conv group 1
    norm1 = tf.layers.batch_normalization(input_)
    conv1 = tf.layers.separable_conv2d(norm1,
                                       32,
                                       3,
                                       strides=1,
                                       padding="same",
                                       activation=params["activation"],
                                       use_bias=params["use_bias"],
                                       pointwise_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       depthwise_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       bias_regularizer=tf.keras.regularizers.l2(weight_decay)
                                       )
    dropout1 = tf.layers.dropout(conv1, rate=0.2)
    pool1 = tf.layers.max_pooling2d(dropout1, 2, 2)

    # conv group 2
    norm2 = tf.layers.batch_normalization(pool1)
    conv2 = tf.layers.conv2d(norm2,
                             64,
                             3,
                             strides=1,
                             padding="same",
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(weight_decay)
                             )
    dropout2 = tf.layers.dropout(conv2, rate=0.3)
    pool2 = tf.layers.average_pooling2d(dropout2, 2, 2)

    # conv group 3
    norm3 = tf.layers.batch_normalization(pool2)
    conv3 = tf.layers.conv2d(norm3,
                             128,
                             3,
                             strides=1,
                             padding="same",
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(weight_decay)
                             )
    dropout3 = tf.layers.dropout(conv3, rate=0.4)
    pool3 = tf.layers.average_pooling2d(dropout3, 2, 2)

    #conv group 4
    norm4 = tf.layers.batch_normalization(pool3)
    conv4 = tf.layers.conv2d(norm4,
                             256,
                             1,
                             strides=1,
                             padding="same",
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                             )
    dropout4 = tf.layers.dropout(conv4, rate=0.4)

    flat = tf.layers.flatten(dropout4)

    out = tf.layers.dense(flat,
                          units=10,
                          activation=None,
                          use_bias=params["use_bias"],
                          )

    pred_class = tf.argmax(out, 1)
    predictions = {
        'class_ids': pred_class[:, tf.newaxis],
        'probabilities': tf.nn.softmax(out),
        'logits': out
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=out)

    loss_mean = tf.reduce_mean(loss)

    pred_one_hot = tf.reshape(tf.one_hot([pred_class], depth=10, on_value=1, off_value=0), params["labels_shape"])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_one_hot,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    opt = tf.train.MomentumOptimizer(params["learning_rate"],
                                     params["momentum"])

    train_op = opt.minimize(loss_mean, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_mean,
            train_op=train_op,
            eval_metric_ops=metrics
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_mean,
            eval_metric_ops=metrics
        )


TRAIN_BATCHSIZE = 100
EVAL_BATCHSIZE = 10000
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 10000
TRAIN_ITER_SIZE = 10000


def cnn_train_input_fn(to_features, to_labels, batch_size=100):
    features = tf.data.Dataset.from_tensor_slices(to_features).map(lambda x: x / 255)
    one_hot = tf.one_hot(list(range(10)), depth=10, on_value=1, off_value=0)
    labels = tf.data.Dataset.from_tensor_slices(to_labels).map(lambda x: one_hot[x, :])
    temp_dataset = tf.data.Dataset.zip((features, labels)).shuffle(TRAIN_SET_SIZE).repeat().batch(batch_size)
    return temp_dataset


def cnn_eval_input_fn(to_features, to_labels):
    features = tf.data.Dataset.from_tensor_slices(to_features).map(lambda x: x / 255)
    one_hot = tf.one_hot(list(range(10)), depth=10, on_value=1, off_value=0)
    labels = tf.data.Dataset.from_tensor_slices(to_labels).map(lambda x: one_hot[x, :])
    temp_dataset = tf.data.Dataset.zip((features, labels))
    return temp_dataset


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, names = import_cifar(data_path)
    temp_dir = os.path.expanduser("~/Desktop/cnn_test")
    params = {
        "activation": tf.nn.leaky_relu,
        "use_bias": True,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "logdir": temp_dir,
        "features_shape": (-1, 32, 32, 3),
        "labels_shape": (-1, 10)
    }
    cnn = tf.estimator.Estimator(cnn_model_fn,
                                 model_dir=temp_dir,
                                 params=params)

    for i in range(1):
        cnn.train(lambda: cnn_train_input_fn(train_x, train_y), steps=1000)
        cnn.evaluate(lambda: cnn_eval_input_fn(test_x, test_y))
    preds = cnn.predict(lambda: cnn_eval_input_fn(test_x, test_y), predict_keys='class_ids')
    for this in preds:
        print(names[this['class_ids'][0]])
        break
