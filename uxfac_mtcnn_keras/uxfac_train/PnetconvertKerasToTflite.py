import tensorflow as tf
import numpy as np
import cv2

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
my_adam = tf.keras.optimizers.Adam(lr=0.00001)

model = tf.keras.Model([input], [classifier, bbox_regress])
model.load_weights('/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/uxfac_model12.h5', by_name=True)

file_path = '/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/12x12_hw_img.txt'
file = open(file_path, 'r')
foo = np.reshape(file.read().split(), newshape=(1, 12, 12))
img = np.zeros((1,12,12,3))
img[:,:,:,0] = foo
img[:,:,:,1] = foo
img[:,:,:,2] = foo

# dummy_input = tf.random.uniform((1, 72, 144, 3))
dummy_input = tf.random.uniform((1, 12, 12, 3))
# dummy_input = img
dummy_output = model.predict(dummy_input)
print(dummy_output)


def dataset_gen():
    for i in range(100):
        img_input = 2 * np.random.rand(1, 12, 12, 3) - 1
        yield [img_input.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = dataset_gen
tflite_quant_model = converter.convert()

with open('uxfac_pnet.tflite', 'wb') as file:
    file.write(tflite_quant_model)
