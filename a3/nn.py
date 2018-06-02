import tensorflow as tf
from typing import Dict
from a3.cifar import import_cifar, prep_dataset, data_path


def nn_model_fn(features: tf.data.Dataset,
                labels: tf.data.Dataset,
                mode: tf.estimator.ModeKeys,
                params: Dict) -> tf.estimator.EstimatorSpec:
    dense1 = tf.layers.dense(features,
                             units=params["d1_units"],
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             # kernel_initializer=params["kernel_init"],
                             # bias_initializer=params["bias_init"],
                             name="dense1")
    dense2 = tf.layers.dense(dense1,
                             units=params["d2_units"],
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             # kernel_initializer=params["kernel_init"],
                             # bias_initializer=params["bias_init"],
                             name="dense2")
    dense3 = tf.layers.dense(dense2,
                             units=params["d3_units"],
                             activation=params["activation"],
                             use_bias=params["use_bias"],
                             # kernel_initializer=params["kernel_init"],
                             # bias_initializer=params["bias_init"],
                             name="dense3")
    out = tf.layers.dense(dense3,
                          units=10,
                          activation=tf.nn.softmax,
                          use_bias=params["use_bias"],
                          # kernel_initializer=params["kernel_init"],
                          # bias_initializer=params["bias_init"],
                          name="out")
    out_reshape = tf.reshape(out, (-1, 1))
    loss = tf.losses.sparse_softmax_cross_entropy(labels, out_reshape)
    # loss_hook = tf.train.SummarySaverHook(
    #     save_steps=100,
    #     output_dir=params["logdir"],
    #     summary_op=loss
    # )
    opt = tf.train.MomentumOptimizer(params["learning_rate"],
                                     params["momentum"])
    train_op = opt.minimize(loss)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=out,
        loss=loss,
        train_op=train_op,
        # training_hooks=[loss_hook]
    )


BATCHSIZE = 128


def nn_train_input_fn() -> tf.data.Dataset:
    train_x, train_y, _, _, _ = import_cifar(data_path)
    train_x, train_y = prep_dataset(train_x, train_y, bReshape=False)
    train_x = train_x.map(lambda x: tf.reshape(x, (-1, 3072)))
    train_y = train_y.map(lambda y: tf.reshape(y, (-1, 1)))
    temp_dataset = tf.data.Dataset.zip((train_x, train_y))
    temp_dataset.shuffle(-1)
    temp_dataset.batch(BATCHSIZE)
    # features = temp_dataset.map(lambda x, y: x)
    # labels = temp_dataset.map(lambda x, y: y)
    return temp_dataset


def nn_eval_input_fn() -> tf.data.Dataset:
    _, _, test_x, test_y, _ = import_cifar(data_path)
    test_x, test_y = prep_dataset(test_x, test_y, bReshape=False)
    return tf.data.Dataset.zip((test_x, test_y))


if __name__ == "__main__":
    temp_dir = "../tmp/nn_test"
    params = {
        "d1_units": 256,
        "d2_units": 512,
        "d3_units": 64,
        "activation": tf.nn.leaky_relu,
        "use_bias": True,
        "kernel_init": tf.initializers.random_normal(0, 0.05),
        "bias_init": tf.initializers.random_normal(0, 0.05),
        "learning_rate": 0.001,
        "momentum": 0.0005,
        "logdir": temp_dir
    }
    nn = tf.estimator.Estimator(nn_model_fn,
                                model_dir=temp_dir,
                                params=params)

    nn.train(nn_train_input_fn, steps=100)

    # train = nn.train(nn_train_input_fn, steps=100)
    #
    # eval = nn.evaluate(nn_eval_input_fn)
    #
    # _, _, _, _, names = import_cifar(data_path)
    # pred = nn.predict(nn_eval_input_fn, predict_keys=names)
    #
    # writer = tf.summary.FileWriter(temp_dir)
    # loss_sum = tf.summary.scalar("loss_sum", "loss")
    # writer.add_summary(loss_sum)
    #
    # init_global = tf.global_variables_initializer()
    # init_local = tf.local_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_global)
    #     sess.run(init_local)
    #     count = 0
    #     while count < 10000:
    #         while True:
    #             count += 1
    #             try:
    #                 sess.run(train)
    #             except:
    #                 # sess.run(eval)
    #                 break
    #     # sess.run(pred)
