import tensorflow as tf


def uxfac12NetModel():
    input = tf.keras.Input(shape=[12, 12, 3])
    x = tf.keras.layers.Conv2D(8, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    classifier = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax', name='classifier1')(x)
    classifier = tf.keras.layers.Reshape((2,))(classifier)
    bbox_regress = tf.keras.layers.Conv2D(4, (1, 1), name='bbox1')(x)
    bbox_regress = tf.keras.layers.Reshape((4,))(bbox_regress)

    model = tf.keras.Model([input], [classifier, bbox_regress])
    model.load_weights('/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/uxfac_model12.h5', by_name=True)

    print(model.summary())


def uxfac24NetModel():
    input = tf.keras.Input(shape=[24, 24, 3])
    x = tf.keras.layers.Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    c1out = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)

    c2input = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(c1out)

    x = tf.keras.layers.Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(c2input)
    c2out = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)

    c3input = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(c2out)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(c3input)
    c3out = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = tf.keras.layers.Flatten()(c3out)
    x = tf.keras.layers.Dense(128, name='dense1')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1], name='prelu4')(x)
    classifier = tf.keras.layers.Dense(2, activation='softmax', name='classifier1')(x)
    bbox_regress = tf.keras.layers.Dense(4, name='bbox1')(x)

    model = tf.keras.Model([input], [classifier, bbox_regress])
    model.load_weights('/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/uxfac_model24.h5', by_name=True)

    print(model.summary())


uxfac12NetModel()
