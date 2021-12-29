import numpy as np
import tensorflow as tf

pnet = []
rnet = []

interpreter = tf.lite.Interpreter(model_path="/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/uxfac_pnet.tflite")
# interpreter = tf.lite.Interpreter(model_path="/home/hong/Downloads/testResultData/test.jpg_0.5.npy")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
all_layers_details = interpreter.get_tensor_details()
f = open('pnetQuantWeight.txt', 'w')
for val in all_layers_details[5:18]:
# for val in all_layers_details:
    # print('-----check-----')
    try:
        data = interpreter.get_tensor(val['index'])
        # print('{}\t{}\t{}'.format(str(val['index']), str(val['name']), str(val['shape'])))
        f.write(str(val) + '\n')
        if val['index'] in [5,8,11,14,16]:
            data = np.transpose(data, (2, 1, 3, 0))
            pnet.append(data)
            for inchannel in data:
                for outchannel in inchannel:
                    f.write(str(outchannel)+'\n')
                f.write('\n')
        elif val['index'] in [7, 10, 13]:
            # data = data.flatten()
            pnet.append(data)
            f.write(str(data) + '\t')
        else:
            pnet.append(data)
            f.write(str(data) + '\t')
        f.write('\n\n')
    except:
        # data = interpreter.get_tensor(val['index'])
        # print('except')
        f.write(str(val) + '\n')
        # f.write(str(data) + '\t')
        # print(str(val))
f.close()

# interpreter = tf.lite.Interpreter(model_path="/home/hong/Documents/python-mtcnn/uxfac_mtcnn_keras/uxfac_train/uxfac_pnet_2.tflite")
# interpreter.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# all_layers_details = interpreter.get_tensor_details()
#
# f = open('pnetQuantWeight_2.txt', 'w')
# for val in all_layers_details:
#     # print('-----check-----')
#     try:
#         data = interpreter.get_tensor(val['index'])
#         # print('{}\t{}\t{}'.format(str(val['index']), str(val['name']), str(val['shape'])))
#         f.write(str(val) + '\n')
#         if val['index'] in [5,8,11,14,16]:
#             data = np.transpose(data, (2, 1, 3, 0))
#             pnet.append(data)
#             for inchannel in data:
#                 for outchannel in inchannel:
#                     f.write(str(outchannel)+'\n')
#                 f.write('\n')
#         elif val['index'] in [7, 10, 13]:
#             # data = data.flatten()
#             pnet.append(data)
#             f.write(str(data) + '\t')
#         else:
#             pnet.append(data)
#             f.write(str(data) + '\t')
#         f.write('\n\n')
#     except:
#         # data = interpreter.get_tensor(val['index'])
#         # print('except')
#         f.write(str(val) + '\n')
#         # f.write(str(data) + '\t')
#         # print(str(val))
# f.close()


interpreter = tf.lite.Interpreter(model_path="uxfac_rnet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
all_layers_details = interpreter.get_tensor_details()
f = open('rnetQuantWeight_2.txt', 'w')
for val in all_layers_details[2:18]:
    # print('-----check-----')
    try:
        data = interpreter.get_tensor(val['index'])
        # print(data.shape)
        # print('{}\t{}\t{}'.format(str(val['index']), str(val['name']), str(val['shape'])))
        f.write(str(val) + '\n')
        # print(str(val))
        if val['index'] in [2, 5, 8]:
            data = np.transpose(data, (2, 1, 3, 0))
            rnet.append(data)
            for inchannel in data:
                for outchannel in inchannel:
                    f.write(str(outchannel)+'\n')
                f.write('\n')
        elif val['index'] in [4, 7, 10]:
            # data = data.flatten()
            rnet.append(data)
            f.write(str(data) + '\t')
        elif val['index'] in [11, 14, 16]:
            # print(np.shape(data))
            rnet.append(data)
            f.write('[')
            for outchannel in data:
                f.write('[')
                for item in outchannel:
                    f.write('{:>3d}\t'.format(item))
                f.write(']\n')
            f.write(']')
        #  shape 에러로 인해 13번 인덱스(prelu4) <-> 28번 인덱스(prelu4-1)=?계산값?
        elif val['index'] == 13:
            data = interpreter.get_tensor(28)
            # data = data.flatten()
            rnet.append(data)
        else:
            rnet.append(data)
            f.write(str(data) + '\t')
        f.write('\n\n')
    except:
        print('except')
        f.write(str(val) + '\n')

weights = {}
pnet[9], pnet[10], pnet[11], pnet[12] = pnet[11], pnet[12], pnet[9], pnet[10]
rnet[12], rnet[13], rnet[14], rnet[15] = np.swapaxes(rnet[14], 0, 1), rnet[15], np.swapaxes(rnet[12], 0, 1), rnet[13]
rnet[9], rnet[11] = np.swapaxes(rnet[9], 0, 1), np.reshape(rnet[11], (128,))
weights['pnet'] = pnet
weights['rnet'] = rnet
for item in pnet:
    print(item.shape)
for item in rnet:
    print(item.shape)

np.save('uxfac_weights_fix_shape', weights)
f.close()

# import tensorflow as tf
#
# interpreter = tf.lite.Interpreter(model_path="uxfac_pnet_2.tflite")
# interpreter.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# all_layers_details = interpreter.get_tensor_details()
# f = open('pnetQuantWeight_2.txt', 'w')
# for val in all_layers_details:
#     print('-----check-----')
#     try:
#         data = interpreter.get_tensor(val['index'])
#         print(str(val))
#         print(str(data))
#         f.write(str(val) + '\n')
#         f.write(str(data) + '\n')
#     except:
#         print('except')
#         f.write(str(val) + '\n')
#         print(str(val))
#
# f.close()