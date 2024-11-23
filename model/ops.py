# -*- coding: utf-8 -*-
import tensorflow as tf

def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # 입력 텐서의 형태 가져오기
        shape = x.get_shape().as_list()
        n_out = shape[-1]

        # 이동평균과 분산을 저장할 변수들 생성
        beta = tf.compat.v1.get_variable('beta', [n_out],
                                        initializer=tf.compat.v1.zeros_initializer())
        gamma = tf.compat.v1.get_variable('gamma', [n_out],
                                         initializer=tf.compat.v1.ones_initializer())
        moving_mean = tf.compat.v1.get_variable('moving_mean', [n_out],
                                               initializer=tf.compat.v1.zeros_initializer(),
                                               trainable=False)
        moving_var = tf.compat.v1.get_variable('moving_var', [n_out],
                                              initializer=tf.compat.v1.ones_initializer(),
                                              trainable=False)

        # 현재 배치의 평균과 분산 계산
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)

        def train_mode():
            # 이동평균 업데이트
            update_mean = tf.compat.v1.assign(moving_mean,
                                            moving_mean * decay + batch_mean * (1 - decay))
            update_var = tf.compat.v1.assign(moving_var,
                                           moving_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

        def test_mode():
            return tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma, epsilon)

        return tf.cond(tf.cast(is_training, tf.bool), train_mode, test_mode)

def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable('W', [kh, kw, shape[-1], output_filters],
                                      initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(x, filters=W, strides=[1, sh, sw, 1], padding='SAME')
        biases = tf.compat.v1.get_variable('b', [output_filters],
                                          initializer=tf.compat.v1.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())
        return Wconv_plus_b

def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        input_shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                                      initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1])
        biases = tf.compat.v1.get_variable('b', [output_shape[-1]],
                                          initializer=tf.compat.v1.constant_initializer(0.0))
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv_plus_b

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable("W", [shape[1], output_size], tf.float32,
                                      initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
        b = tf.compat.v1.get_variable("b", [output_size],
                                      initializer=tf.compat.v1.constant_initializer(0.0))
        return tf.matmul(x, W) + b

def init_embedding(size, dimension, stddev=0.01, scope="embedding"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        return tf.compat.v1.get_variable("E", [size, 1, 1, dimension], tf.float32,
                                         initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))

def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        shape = x.get_shape().as_list()
        batch_size, output_filters = shape[0], shape[-1]

        scale = tf.compat.v1.get_variable("scale", [labels_num, output_filters], tf.float32,
                                          initializer=tf.compat.v1.constant_initializer(1.0))
        shift = tf.compat.v1.get_variable("shift", [labels_num, output_filters], tf.float32,
                                          initializer=tf.compat.v1.constant_initializer(0.0))

        # Instance Normalization
        mu, sigma = tf.nn.moments(x, [1, 2], keepdims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)

        # Conditional scaling and shifting
        batch_scale = tf.reshape(tf.nn.embedding_lookup(params=scale, ids=ids),
                                 [batch_size, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup(params=shift, ids=ids),
                                 [batch_size, 1, 1, output_filters])

        return norm * batch_scale + batch_shift