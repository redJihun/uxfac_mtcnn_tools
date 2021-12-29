import numpy as np

# data = np.load('/home/hong/Documents/python-mtcnn/mtcnn_weights.npy', allow_pickle=True).tolist()
# # print(data['pnet'])
# for array in data['pnet']:
#     print(array)
#     print(np.shape(array))
#
# for item in data:
#     for it in item:
#         print(it)
def load_sram(file_path):
    weight_list = []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        # for line in lines:
        for line in lines:
            reverse_list = []
            for i in range(16):
                reverse_list.append(line[i * 2:i * 2 + 2])
            reverse_list.reverse()
            reverse_list = [int(item, 16) for item in reverse_list]
            for i in range(len(reverse_list)):
                if reverse_list[i] > 127:
                    reverse_list[i] = reverse_list[i] - 256
                weight_list.append(reverse_list[i])
        # print(np.shape(weight_list))

    return weight_list


# def init_weight(sram4, sram5, sram6):
#     pnet = []
#     conv1[3][3][3][10]
#     rnet = []
#     offset4, offset5, offset6 = 0, 0, 0
#     for i in range(3):
#
#
#     return test



if __name__ == '__main__':
    sram4 = load_sram('/home/hong/Documents/python-mtcnn/211115_sram4_test2.coe')
    sram5 = load_sram('/home/hong/Documents/python-mtcnn/211115_sram5_test2.coe')
    sram6 = load_sram('/home/hong/Documents/python-mtcnn/211115_sram6_test2.coe')
    pnet = []
    rnet = []
    print('4: {}\n5: {}\n6: {}'.format(np.shape(sram4), np.shape(sram5), np.shape(sram6)))
    data = np.load('/check_weight/mtcnn_weights.npy', allow_pickle=True).tolist()
    # conv1
    print('conv1: {}'.format(np.shape(data['pnet'][0])))
    foo = np.reshape(sram4[:216], newshape=[3, 8, 3, 3])
    foo = np.swapaxes(foo, 0, 2)
    foo = np.swapaxes(foo, 1, 3)
    foo = np.swapaxes(foo, 0, 1)
    pnet.append(foo)
    print('pnet-conv1: {}'.format(np.shape(pnet[0])))
    # bias1
    print('bias1: {}'.format(np.shape(data['pnet'][1])))
    # act1
    print('act1: {}'.format(np.shape(data['pnet'][2])))
    # conv2
    # print('conv2: {}'.format(np.shape(data['pnet'][3])))
    # foo = np.reshape(sram4[224:1376], newshape=[8, 16, 3, 3])
    # foo = np.swapaxes(foo, 0, 2)
    # foo = np.swapaxes(foo, 1, 3)
    # foo = np.swapaxes(foo, 0, 1)
    # pnet.append(foo)
    # print('pnet-conv2: {}'.format(np.shape(pnet[3])))
    # # bias2
    # print('bias2: {}'.format(np.shape(data['pnet'][4])))
    # # act2
    # print('act2: {}'.format(np.shape(data['pnet'][5])))
    # # conv3
    # print('conv3: {}'.format(np.shape(data['pnet'][6])))
    # foo = np.reshape(sram4[1376:5984], newshape=[16, 32, 3, 3])
    # foo = np.swapaxes(foo, 0, 2)
    # foo = np.swapaxes(foo, 1, 3)
    # foo = np.swapaxes(foo, 0, 1)
    # pnet.append(foo)
    # # bias3
    # print('bias3: {}'.format(np.shape(data['pnet'][7])))
    # # act3
    # print('act3: {}'.format(np.shape(data['pnet'][8])))
    # # conv4
    # print('conv4: {}'.format(np.shape(data['pnet'][9])))
    # foo = np.reshape(sram4[5984:6048], newshape=[32, 2, 1, 1])
    # foo = np.swapaxes(foo, 0, 2)
    # foo = np.swapaxes(foo, 1, 3)
    # foo = np.swapaxes(foo, 0, 1)
    # pnet.append(foo)
    # # bias4
    # print('bias4: {}'.format(np.shape(data['pnet'][10])))
    # # conv5
    # print('conv5: {}'.format(np.shape(data['pnet'][11])))
    # foo = np.reshape(sram4[6048:6176], newshape=[32, 4, 1, 1])
    # foo = np.swapaxes(foo, 0, 2)
    # foo = np.swapaxes(foo, 1, 3)
    # foo = np.swapaxes(foo, 0, 1)
    # pnet.append(foo)
    # # bias5
    # print('bias5: {}'.format(np.shape(data['pnet'][12])))
    # print(np.shape(pnet[4]))



