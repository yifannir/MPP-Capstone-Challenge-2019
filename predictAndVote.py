import sklearn

import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 128*173
labelsPic_train = pd.read_csv('speech-spectrograms/train_labels.csv')
listpic_train = []
listlabel_train = []

listpic_eval = []
listlabel_eval = []

for i in range(len(labelsPic_train)):
    picName = 'speech-spectrograms/train/' + str(labelsPic_train.iloc[i, 0]) + '.png'
    if i < 1 * len(labelsPic_train):

        listpic_train.append(picName)
        listlabel_train.append(labelsPic_train.iloc[i, 1])
    else:
        listpic_eval.append(picName)
        listlabel_eval.append(labelsPic_train.iloc[i, 1])

features = tf.constant(listpic_train)
labels = tf.constant(listlabel_train)


def _parse_function(picPath, label):
    image_string = tf.read_file(picPath)
    image_decoded = tf.image.decode_png(image_string)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)

    image_decoded = tf.image.random_crop(image_decoded, [128, 128, 1])
    image_decoded = tf.image.random_flip_left_right(image_decoded)
    # image_decoded = tf.image.random_brightness(image_decoded,0.1)
    return image_decoded, tf.one_hot(label, depth=3)


def _parse_function_predict(picPath):
    image_string = tf.read_file(picPath)
    image_decoded = tf.image.decode_png(image_string)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = tf.image.random_crop(image_decoded, [128, 128, 1])
    return image_decoded


def train_input_fn(batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.map(_parse_function)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(4000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def train_input_fn2(batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.map(_parse_function)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(4000).batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


features_eval = tf.constant(listpic_eval)
labels_eval = tf.constant(listlabel_eval)


def eval_input_fn(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features_eval, labels_eval))
    dataset = dataset.map(_parse_function)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(4000).batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn2(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features_eval, labels_eval))
    dataset = dataset.map(_parse_function)
    # Shuffle, repeat, and batch the examples.
    # Return the read end of the pipeline.
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


labelsPic_predict = pd.read_csv('speech-spectrograms/submission_format.csv')
listpic_predict = []
for i in range(len(labelsPic_predict)):
    picName = 'speech-spectrograms/test/' + str(labelsPic_predict.iloc[i, 0]) + '.png'
    listpic_predict.append(picName)

features_predict = tf.constant(listpic_predict)


def predict_input_fn(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features_predict))
    dataset = dataset.map(_parse_function_predict)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def cnn_model(features, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope('Input'):
        # Input Layer
        input_layer = tf.reshape(features, [-1, 128, 128, 1], name='input_reshape')

        tf.summary.image('input', input_layer)
        input_layer = tf.layers.batch_normalization(input_layer)
    with tf.name_scope('Conv_1'):
        # Convolutional Layer #1
        conv1_1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=8,
            kernel_size=(2, 2),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)
        conv1_2 = tf.layers.conv2d(
            inputs=conv1_1,
            filters=8,
            kernel_size=(2, 2),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)

        # Pooling Layer #1
        pool1 = tf.layers.average_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2, padding='same')

    with tf.name_scope('Conv_2'):
        # Convolutional Layer #2 and Pooling Layer #2
        pool1 = tf.layers.batch_normalization(pool1)
        conv2_1 = tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)
        conv2_2 = tf.layers.conv2d(
            inputs=conv2_1,
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)

        # Pooling Layer #1
        pool2 = tf.layers.average_pooling2d(inputs=conv2_2, pool_size=(2, 2), strides=2, padding='same')

    with tf.name_scope('Conv_3'):
        # Convolutional Layer #2 and Pooling Layer #2
        pool2 = tf.layers.batch_normalization(pool2)
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)

        pool3 = tf.layers.average_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2, padding='same')

    with tf.name_scope('Dense_Dropout'):
        # Dense Layer
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        pool3_flat = tf.contrib.layers.flatten(pool3)
        pool3_flat = tf.layers.batch_normalization(pool3_flat)
        dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu, trainable=is_training)
        dropout = tf.layers.dropout(inputs=dense, rate=params['dropout_rate'], training=is_training)

    with tf.name_scope('Predictions'):
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=3, trainable=is_training)

        return logits


def my_model(features, labels, mode, params):
    """Model function for CNN."""

    logits = cnn_model(features, mode, params)
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    scores = tf.nn.softmax(logits, name='softmax_tensor')
    # Generate Predictions
    predictions = {
        'classes': predicted_logit,
        'probabilities': scores
    }

    export_outputs = {
        'prediction': tf.estimator.export.ClassificationOutput(
            scores=scores,
            classes=tf.cast(predicted_logit, tf.string))
    }

    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # TRAIN and EVAL
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predicted_logit)
    eval_metric = {'accuracy': accuracy}

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('accuracy', accuracy[0])
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params['learning_rate'],
            optimizer='Adam')
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric,
        predictions=predictions,
        export_outputs=export_outputs)


def getmax(nums):
    counts = np.bincount(nums)
    return np.argmax(counts)


with tf.Session() as sess:
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'learning_rate': 0.00002,
            'dropout_rate': 0.5,
            'n_classes': 3,
        }, model_dir='modelExp/Cnn3_bn2')

    # # anylise
    # anylise = list(classifier.predict(input_fn=lambda: eval_input_fn2(1), predict_keys='classes'))
    # anyliseC = []
    # for result in anylise:
    #     anyliseC.append(result['classes'])
    # confuse = sklearn.metrics.confusion_matrix(listlabel_eval,anyliseC)
    # print(confuse)
    '''
    predict
    '''
    lunshu = 1314
    ansList = []
    for i in range(lunshu):
        print(i)
        predict_result = list(classifier.predict(input_fn=lambda: predict_input_fn(400), predict_keys='classes'))
        ans = []
        for result in predict_result:
            ans.append(result['classes'])

        ansList.append(ans)
    finalAns = []
    for i in range(len(ansList[0])):
        tmp = []
        for j in range(lunshu):
            tmp.append(ansList[j][i])
        finalAns.append(getmax(tmp))

    for i in range(len(labelsPic_predict)):
        labelsPic_predict.iloc[i, 1] = finalAns[i]
    labelsPic_predict.to_csv('sub30_1314.csv', index=None)
    print('finish')
