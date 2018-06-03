import tensorflow as tf
from typing import Dict
from a3.cifar import import_cifar, prep_dataset, data_path


def nn_model_fn(features: tf.data.Dataset,
                labels: tf.data.Dataset,
                mode: tf.estimator.ModeKeys,
                params: Dict) -> tf.estimator.EstimatorSpec:
    # input = tf.layers.InputSpec(dtype=tf.float32, shape=(-1, 3072))
    dense1 = tf.layers.dense(features,
                             units=params["d1_units"],
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             # kernel_initializer=params["kernel_init"],
                             # bias_initializer=params["bias_init"],
                             # name="dense1"
                             )
    dense2 = tf.layers.dense(dense1,
                             units=params["d2_units"],
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             # kernel_initializer=params["kernel_init"],
                             # bias_initializer=params["bias_init"],
                             # name="dense2"
                             )
    dense3 = tf.layers.dense(dense2,
                             units=params["d3_units"],
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             # kernel_initializer=params["kernel_init"],
                             # bias_initializer=params["bias_init"],
                             # name="dense3"
                             )
    out = tf.layers.dense(dense3,
                          units=10,
                          activation=None,
                          use_bias=params["use_bias"],
                          # kernel_initializer=params["kernel_init"],
                          # bias_initializer=params["bias_init"],
                          # name="out"
                          )

    pred_class = tf.argmax(out, 1)
    predictions = {
        'class_ids': pred_class[:, tf.newaxis],
        'probabilities': tf.nn.softmax(out),
        'logits': out,
    }

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=out)

    pred_one_hot = tf.reshape(tf.one_hot([pred_class], depth=10, on_value=1, off_value=0), (-1,))

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_one_hot,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    opt = tf.train.MomentumOptimizer(params["learning_rate"],
                                     params["momentum"])

    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )


BATCHSIZE = 2048


def nn_train_input_fn() -> tf.data.Dataset:
    train_x, train_y, _, _, _ = import_cifar(data_path)
    train_x, train_y = prep_dataset(train_x, train_y, bReshape=False)
    train_x = train_x.map(lambda x: tf.reshape(x, (-1, 3072)))
    temp_dataset = tf.data.Dataset.zip((train_x, train_y))
    temp_dataset.shuffle(-1)
    temp_dataset.batch(BATCHSIZE)
    return temp_dataset


def nn_eval_input_fn() -> tf.data.Dataset:
    _, _, test_x, test_y, _ = import_cifar(data_path)
    test_x, test_y = prep_dataset(test_x, test_y, bReshape=False)
    test_x = test_x.map(lambda x: tf.reshape(x, (-1, 3072)))
    return tf.data.Dataset.zip((test_x, test_y))


if __name__ == "__main__":
    _, _, _, _, names = import_cifar(data_path)
    temp_dir = "../tmp/nn_test"
    params = {
        "d1_units": 256,
        "d2_units": 512,
        "d3_units": 64,
        "activation": tf.nn.leaky_relu,
        "use_bias": True,
        # "kernel_init": tf.initializers.random_normal(0, 0.05),
        # "bias_init": tf.initializers.random_normal(0, 0.05),
        "learning_rate": 0.001,
        "momentum": 0.9,
        "logdir": temp_dir
    }
    nn = tf.estimator.Estimator(nn_model_fn,
                                model_dir=temp_dir,
                                params=params)

    for i in range(10):
        nn.train(nn_train_input_fn, steps=5000)
        nn.evaluate(nn_eval_input_fn)
    preds = nn.predict(nn_eval_input_fn, predict_keys=names)
