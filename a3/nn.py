import tensorflow as tf
from typing import Dict
from a3.cifar import import_cifar, data_path
import os


def nn_model_fn(features: tf.data.Dataset,
                labels: tf.data.Dataset,
                mode: tf.estimator.ModeKeys,
                params: Dict) -> tf.estimator.EstimatorSpec:
    input_ = tf.reshape(features, params["features_shape"])
    if labels is not None:
        labels = tf.reshape(labels, params["labels_shape"])

    weight_decay = 1e-4

    dropout = input_

    for units in params["units"]:
        dense = tf.layers.dense(dropout,
                                units,
                                activation=params["activation"],
                                use_bias=params["use_bias"],
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                bias_regularizer=tf.keras.regularizers.l2(weight_decay)
                                )
        norm = tf.layers.batch_normalization(dense)
        dropout = tf.layers.dropout(norm)

    out = tf.layers.dense(dropout,
                          units=10,
                          activation=None,
                          use_bias=params["use_bias"],
                          )

    pred_class = tf.argmax(out, 1)
    predictions = {
        'class_ids': pred_class[:, tf.newaxis],
        'probabilities': tf.nn.softmax(out),
        'logits': out,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
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
            # train_op=train_op,
            eval_metric_ops=metrics
        )


TRAIN_BATCHSIZE = 100
EVAL_BATCHSIZE = 10000
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 10000
TRAIN_ITER_SIZE = 10000


def nn_train_input_fn(to_features, to_labels, batch_size=100):
    features = tf.data.Dataset.from_tensor_slices(to_features).map(lambda x: x / 255)
    one_hot = tf.one_hot(list(range(10)), depth=10, on_value=1, off_value=0)
    labels = tf.data.Dataset.from_tensor_slices(to_labels).map(lambda x: one_hot[x, :])
    temp_dataset = tf.data.Dataset.zip((features, labels)).shuffle(TRAIN_SET_SIZE).repeat().batch(batch_size)
    return temp_dataset


def nn_eval_input_fn(to_features, to_labels):
    features = tf.data.Dataset.from_tensor_slices(to_features).map(lambda x: x / 255)
    one_hot = tf.one_hot(list(range(10)), depth=10, on_value=1, off_value=0)
    labels = tf.data.Dataset.from_tensor_slices(to_labels).map(lambda x: one_hot[x, :])
    temp_dataset = tf.data.Dataset.zip((features, labels))
    return temp_dataset


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, names = import_cifar(data_path)
    temp_dir = os.path.expanduser("~/Desktop/nn_test")
    params = {
        "units": [2048, 2048, 2048],
        "activation": tf.nn.leaky_relu,
        "use_bias": True,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "logdir": temp_dir,
        "features_shape": (-1, 3072),
        "labels_shape": (-1, 10)
    }
    nn = tf.estimator.Estimator(nn_model_fn,
                                model_dir=temp_dir,
                                params=params)

    for i in range(6):
        nn.train(lambda: nn_train_input_fn(train_x, train_y), steps=1000)
        nn.evaluate(lambda: nn_eval_input_fn(test_x, test_y))
    preds = nn.predict(lambda: nn_eval_input_fn(test_x, test_y))
    for this in preds:
        print(repr(this))
        break
