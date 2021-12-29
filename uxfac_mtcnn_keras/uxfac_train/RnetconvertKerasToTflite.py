import tensorflow as tf
import numpy as np

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

my_adam = tf.keras.optimizers.Adam(lr=0.001)

model = tf.keras.Model([input], [classifier, bbox_regress])
model.load_weights('/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/uxfac_model24.h5', by_name=True)

dummy_input = tf.random.uniform((1, 24, 24, 3))
dummy_output = model.predict(dummy_input)


def dataset_gen():
    for i in range(100):
        img_input = 2 * np.random.rand(1, 24, 24, 3) - 1
        yield [img_input.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = dataset_gen
tflite_quant_model = converter.convert()

with open('uxfac_rnet.tflite', 'wb') as file:
    file.write(tflite_quant_model)
