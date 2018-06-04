import tensorflow as tf
from typing import Dict
from a3.cifar import import_cifar, prep_dataset, data_path


def cnn_model_fn(features: tf.Tensor,
                 labels: tf.Tensor,
                 mode: tf.estimator.ModeKeys,
                 params: Dict) -> tf.estimator.EstimatorSpec:

    weight_decay = 1e-4

    input_ = tf.reshape(features, (-1, 32, 32, 3))

    if labels is not None:
        labels = tf.reshape(labels, (-1, 10))

    # conv group 1
    conv1 = tf.layers.separable_conv2d(input_,
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
    norm1 = tf.layers.batch_normalization(conv1)
    dropout1 = tf.layers.dropout(norm1, rate=0.2)

    # conv group 2
    conv2 = tf.layers.conv2d(dropout1,
                             64,
                             3,
                             strides=2,
                             padding="same",
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(weight_decay)
                             )
    norm2 = tf.layers.batch_normalization(conv2)
    dropout2 = tf.layers.dropout(norm2, rate=0.3)

    # conv group 3
    conv3 = tf.layers.conv2d(dropout2,
                             128,
                             3,
                             strides=2,
                             padding="same",
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(weight_decay)
                             )
    norm3 = tf.layers.batch_normalization(conv3)
    dropout3 = tf.layers.dropout(norm3, rate=0.4)

    flat = tf.layers.flatten(dropout3)

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

    # pred_one_hot = tf.reshape(tf.one_hot([pred_class], depth=10, on_value=1, off_value=0), (-1, ))

    pred_one_hot = tf.reshape(tf.one_hot([pred_class], depth=10, on_value=1, off_value=0), (-1, 10))

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


def cnn_train_input_fn() -> tf.data.Dataset:
    train_x, train_y, _, _, _ = import_cifar(data_path)
    train_x, train_y = prep_dataset(train_x, train_y)
    temp_dataset = tf.data.Dataset.zip((train_x, train_y))
    temp_dataset = temp_dataset.shuffle(TRAIN_SET_SIZE).repeat().batch(TRAIN_BATCHSIZE)
    return temp_dataset


def cnn_eval_input_fn() -> tf.data.Dataset:
    _, _, test_x, test_y, _ = import_cifar(data_path)
    test_x, test_y = prep_dataset(test_x, test_y)
    temp_dataset = tf.data.Dataset.zip((test_x, test_y))
    # temp_dataset = temp_dataset.batch(EVAL_BATCHSIZE)
    return temp_dataset


if __name__ == "__main__":
    _, _, _, _, names = import_cifar(data_path)
    temp_dir = "../tmp/cnn_test"
    params = {
        "activation": tf.nn.leaky_relu,
        "use_bias": True,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "logdir": temp_dir,
    }
    cnn = tf.estimator.Estimator(cnn_model_fn,
                                 model_dir=temp_dir,
                                 params=params)

    for i in range(5):
        cnn.train(cnn_train_input_fn, steps=3000)
        cnn.evaluate(cnn_eval_input_fn)
    preds = cnn.predict(cnn_eval_input_fn, predict_keys='class_ids')
    for this in preds:
        print(names[this['class_ids'][0]])
        break
