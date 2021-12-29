import numpy as np


def make_quant_list():
    quant_list = []
    for idx in range(128):
        quant_list.append(-2.0 + (0.03125 * idx))
    return quant_list


def fixed(num):
    fixed_num = num
    # data = data * (2**28)
    # data = int(math.floor(data))
    # return int(hex(data).replace('0x',''), 16)
    return fixed_num


def to_hex(val, nbits):
    return hex((val + (1 << nbits)) % (1 << nbits)).replace('0x', '')


def quant(num, quant_list=make_quant_list()):
    min_value = 99999999
    min_index = 0
    for idx in range(len(quant_list)):
        if min_value > abs(quant_list[idx] - num):
            min_value = min(min_value, abs(quant_list[idx] - num))
            min_index = idx
    return quant_list[min_index]


# data = np.load('/home/hong/Documents/python-mtcnn/mtcnn_weight_coe2npy/converted_npy/12_08_215123_weight.npy', allow_pickle=True).tolist()
data = np.load('/home/hong/Documents/python-mtcnn/mtcnn_weight_coe2npy/converted_npy/p_oc_8.npy', allow_pickle=True).tolist()
sram4 = open('converted_coe/fw5_sram4.txt', 'w')
sram5 = open('converted_coe/fw5_sram5.txt', 'w')
sram6 = open('converted_coe/fw5_sram6.txt', 'w')
pnet, rnet = data['pnet'], data['rnet']

# pnet-conv1 ###########################################################################################################
pnet_conv1 = pnet[0][:, :, :, :8]
pnet_conv1 = np.transpose(pnet_conv1, (2, 3, 1, 0))
pnet_conv1 = pnet_conv1.flatten()
while len(pnet_conv1) < 224:
    pnet_conv1 = np.append(pnet_conv1, 0)
pnet_conv1 = pnet_conv1.reshape(14, 16)
for i in range(len(pnet_conv1)):
    for j in range(len(pnet_conv1[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(pnet_conv1[i][j]))))
    sram4.write('\n')

# pnet-bias1-act1 ######################################################################################################
pnet_bias1 = pnet[1]
pnet_act1 = pnet[2].flatten()
for i in range(7, 3, -1):
    sram6.write('{}\t'.format((quant(pnet_act1[i]))))
for i in range(7, 3, -1):
    sram6.write('{}\t'.format((quant(pnet_bias1[i]))))
for i in range(3, -1, -1):
    sram6.write('{}\t'.format((quant(pnet_act1[i]))))
for i in range(3, -1, -1):
    sram6.write('{}\t'.format((quant(pnet_bias1[i]))))
sram6.write('\n')

# pnet-conv2 ###########################################################################################################
pnet_conv2 = pnet[3][:, :, :8, :]
pnet_conv2 = np.transpose(pnet_conv2, (2, 3, 1, 0))
pnet_conv2 = pnet_conv2.flatten()
while len(pnet_conv2) < 1152:
    pnet_conv2 = np.append(pnet_conv2, 0)
pnet_conv2 = pnet_conv2.reshape(72, 16)
for i in range(len(pnet_conv2)):
    for j in range(len(pnet_conv2[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(pnet_conv2[i][j]))))
    sram4.write('\n')

# pnet-bias2-act2 ######################################################################################################
pnet_bias2 = pnet[4]
pnet_act2 = pnet[5].flatten()
for i in range(7, 3, -1):
    sram6.write('{}\t'.format((quant(pnet_act2[i]))))
for i in range(7, 3, -1):
    sram6.write('{}\t'.format((quant(pnet_bias2[i]))))
for i in range(3, -1, -1):
    sram6.write('{}\t'.format((quant(pnet_act2[i]))))
for i in range(3, -1, -1):
    sram6.write('{}\t'.format((quant(pnet_bias2[i]))))
sram6.write('\n')
for i in range(15, 11, -1):
    sram6.write('{}\t'.format((quant(pnet_act2[i]))))
for i in range(15, 11, -1):
    sram6.write('{}\t'.format((quant(pnet_bias2[i]))))
for i in range(11, 7, -1):
    sram6.write('{}\t'.format((quant(pnet_act2[i]))))
for i in range(11, 7, -1):
    sram6.write('{}\t'.format((quant(pnet_bias2[i]))))
sram6.write('\n')

# pnet-conv3 ###########################################################################################################
pnet_conv3_1 = pnet[6][:, :, :, :16]
pnet_conv3_2 = pnet[6][:, :, :, 16:]
pnet_conv3_1 = np.transpose(pnet_conv3_1, (2, 3, 1, 0))
pnet_conv3_1 = pnet_conv3_1.flatten()
pnet_conv3_2 = np.transpose(pnet_conv3_2, (2, 3, 1, 0))
pnet_conv3_2 = pnet_conv3_2.flatten()
while len(pnet_conv3_1) < 2304:
    pnet_conv3_1 = np.append(pnet_conv3_1, 0)
while len(pnet_conv3_2) < 2304:
    pnet_conv3_2 = np.append(pnet_conv3_2, 0)
pnet_conv3 = np.append(pnet_conv3_1, pnet_conv3_2)
pnet_conv3 = pnet_conv3.reshape(288, 16)

for i in range(len(pnet_conv3)):
    for j in range(len(pnet_conv3[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(pnet_conv3[i][j]))))
    sram4.write('\n')

# pnet-bias3-act3 ######################################################################################################
pnet_bias3 = pnet[7]
pnet_act3 = pnet[8].flatten()

for index in range(4):
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(pnet_act3[i]))))
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(pnet_bias3[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(pnet_act3[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(pnet_bias3[i]))))
    sram6.write('\n')

# pnet-conv4 ###########################################################################################################
pnet_conv4 = pnet[9]
pnet_conv4 = np.transpose(pnet_conv4, (3, 2, 0, 1))
pnet_conv4 = np.swapaxes(pnet_conv4, 2, 3)
pnet_conv4 = np.swapaxes(pnet_conv4, 0, 2)
pnet_conv4 = np.swapaxes(pnet_conv4, 1, 3)
pnet_conv4 = pnet_conv4.flatten()
while len(pnet_conv4) < 64:
    pnet_conv4 = np.append(pnet_conv4, 0)
pnet_conv4 = pnet_conv4.reshape(4, 16)
for i in range(len(pnet_conv4)):
    for j in range(len(pnet_conv4[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(pnet_conv4[i][j]))))
    sram4.write('\n')

# pnet-conv5 ###########################################################################################################
pnet_conv5 = pnet[11]
pnet_conv5 = np.transpose(pnet_conv5, (3, 2, 0, 1))
pnet_conv5 = pnet_conv5.flatten()
while len(pnet_conv5) < 128:
    pnet_conv5 = np.append(pnet_conv5, 0)
pnet_conv5 = pnet_conv5.reshape(8, 16)
for i in range(len(pnet_conv5)):
    for j in range(len(pnet_conv5[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(pnet_conv5[i][j]))))
    sram4.write('\n')

# pnet-bias4-bias5 #####################################################################################################
pnet_bias4 = pnet[10]
pnet_bias5 = pnet[12]
for i in range(10):
    sram6.write('{}\t'.format(0))
for i in range(4 - 1, -1, -1):
    sram6.write('{}\t'.format((quant(pnet_bias5[i]))))
for i in range(2 - 1, -1, -1):
    sram6.write('{}\t'.format((quant(pnet_bias4[i]))))
sram6.write('\n')

# rnet-conv1 ###########################################################################################################
rnet_conv1_1 = rnet[0][:, :, :, :16]
rnet_conv1_2 = rnet[0][:, :, :, 16:]
rnet_conv1_1 = np.transpose(rnet_conv1_1, (2, 3, 1, 0))
rnet_conv1_1 = rnet_conv1_1.flatten()
rnet_conv1_2 = np.transpose(rnet_conv1_2, (2, 3, 1, 0))
rnet_conv1_2 = rnet_conv1_2.flatten()
while len(rnet_conv1_1) < 432:
    rnet_conv1_1 = np.append(rnet_conv1_1, 0)
while len(rnet_conv1_2) < 352:
    rnet_conv1_2 = np.append(rnet_conv1_2, 0)
rnet_conv1 = np.append(rnet_conv1_1, rnet_conv1_2)
rnet_conv1 = rnet_conv1.reshape(49, 16)
for i in range(len(rnet_conv1)):
    for j in range(len(rnet_conv1[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(rnet_conv1[i][j]))))
    sram4.write('\n')

# rnet-bias1-act1 ######################################################################################################
rnet_bias1 = rnet[1]
rnet_act1 = rnet[2].flatten()
for index in range(3):
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_act1[i]))))
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_bias1[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_act1[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_bias1[i]))))
    sram6.write('\n')
for i in range(8):
    sram6.write('{}\t'.format(0))
for i in range(24 - 1, 20 - 1, -1):
    sram6.write('{}\t'.format((quant(rnet_act1[i]))))
for i in range(24 - 1, 20 - 1, -1):
    sram6.write('{}\t'.format((quant(rnet_bias1[i]))))
sram6.write('\n')

# rnet-conv2 ###########################################################################################################
rnet_conv2_1 = rnet[3][:, :, :, :16]
rnet_conv2_2 = rnet[3][:, :, :, 16:32]
rnet_conv2_3 = rnet[3][:, :, :, 32:]
rnet_conv2_1 = np.transpose(rnet_conv2_1, (2, 3, 1, 0))
rnet_conv2_1 = rnet_conv2_1.flatten()
rnet_conv2_2 = np.transpose(rnet_conv2_2, (2, 3, 1, 0))
rnet_conv2_2 = rnet_conv2_2.flatten()
rnet_conv2_3 = np.transpose(rnet_conv2_3, (2, 3, 1, 0))
rnet_conv2_3 = rnet_conv2_3.flatten()
while len(rnet_conv2_1) < 4032:
    rnet_conv2_1 = np.append(rnet_conv2_1, 0)
while len(rnet_conv2_2) < 4032:
    rnet_conv2_2 = np.append(rnet_conv2_2, 0)
while len(rnet_conv2_3) < 4032:
    rnet_conv2_3 = np.append(rnet_conv2_3, 0)
rnet_conv2 = np.append(rnet_conv2_1, rnet_conv2_2)
rnet_conv2 = np.append(rnet_conv2, rnet_conv2_3)
rnet_conv2 = rnet_conv2.reshape(756, 16)
for i in range(len(rnet_conv2)):
    for j in range(len(rnet_conv2[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(rnet_conv2[i][j]))))
    sram4.write('\n')

# rnet-bias2-act2 ######################################################################################################
rnet_bias2 = rnet[4]
rnet_act2 = rnet[5].flatten()
for index in range(6):
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_act2[i]))))
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_bias2[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_act2[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_bias2[i]))))
    sram6.write('\n')

# rnet-conv3 ###########################################################################################################
rnet_conv3_1 = rnet[6][:, :, :, :16]
rnet_conv3_2 = rnet[6][:, :, :, 16:32]
rnet_conv3_3 = rnet[6][:, :, :, 32:48]
rnet_conv3_4 = rnet[6][:, :, :, 48:]
rnet_conv3_1 = np.transpose(rnet_conv3_1, (2, 3, 1, 0))
rnet_conv3_1 = rnet_conv3_1.flatten()
rnet_conv3_2 = np.transpose(rnet_conv3_2, (2, 3, 1, 0))
rnet_conv3_2 = rnet_conv3_2.flatten()
rnet_conv3_3 = np.transpose(rnet_conv3_3, (2, 3, 1, 0))
rnet_conv3_3 = rnet_conv3_3.flatten()
rnet_conv3_4 = np.transpose(rnet_conv3_4, (2, 3, 1, 0))
rnet_conv3_4 = rnet_conv3_4.flatten()
while len(rnet_conv3_1) < 6912:
    rnet_conv3_1 = np.append(rnet_conv3_1, 0)
while len(rnet_conv3_2) < 6912:
    rnet_conv3_2 = np.append(rnet_conv3_2, 0)
while len(rnet_conv3_3) < 6912:
    rnet_conv3_3 = np.append(rnet_conv3_3, 0)
while len(rnet_conv3_4) < 6912:
    rnet_conv3_4 = np.append(rnet_conv3_4, 0)
rnet_conv3 = np.append(rnet_conv3_1, rnet_conv3_2)
rnet_conv3 = np.append(rnet_conv3, rnet_conv3_3)
rnet_conv3 = np.append(rnet_conv3, rnet_conv3_4)
rnet_conv3 = rnet_conv3.reshape(1728, 16)
for i in range(len(rnet_conv3)):
    for j in range(len(rnet_conv3[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(rnet_conv3[i][j]))))
    sram4.write('\n')

# rnet-bias3-act3 ######################################################################################################
rnet_bias3 = rnet[7]
rnet_act3 = rnet[8].flatten()
for index in range(8):
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant((rnet_act3[i])))))
    for i in range(index * 8 + 8 - 1, index * 8 + 4 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_bias3[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_act3[i]))))
    for i in range(index * 8 + 4 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_bias3[i]))))
    sram6.write('\n')

# rnet-fc1 #############################################################################################################
rnet_fc1 = rnet[9]
rnet_fc1 = np.swapaxes(rnet_fc1, 0, 1)
rnet_fc1_temp = np.zeros(rnet_fc1.shape)
for i in range(128):
    for j in range(64):
        rnet_fc1_temp[i][j * 9] = rnet_fc1[i][j]
        rnet_fc1_temp[i][j * 9 + 1] = rnet_fc1[i][j + 64]
        rnet_fc1_temp[i][j * 9 + 2] = rnet_fc1[i][j + 128]
        rnet_fc1_temp[i][j * 9 + 3] = rnet_fc1[i][j + 192]
        rnet_fc1_temp[i][j * 9 + 4] = rnet_fc1[i][j + 256]
        rnet_fc1_temp[i][j * 9 + 5] = rnet_fc1[i][j + 320]
        rnet_fc1_temp[i][j * 9 + 6] = rnet_fc1[i][j + 384]
        rnet_fc1_temp[i][j * 9 + 7] = rnet_fc1[i][j + 448]
        rnet_fc1_temp[i][j * 9 + 8] = rnet_fc1[i][j + 512]
net_fc1 = rnet_fc1_temp.flatten()
rnet_fc1 = rnet_fc1.reshape(4608, 16)
for i in range(len(rnet_fc1)):
    for j in range(len(rnet_fc1[i]) - 1, -1, -1):
        sram5.write('{}\t'.format((quant(rnet_fc1[i][j]))))
    sram5.write('\n')

# rnet-bias4-act4 ######################################################################################################
rnet_bias4 = rnet[10]
rnet_act4 = rnet[11].flatten()
for index in range(16):
    for i in range(index * 8 + 8 - 1, index * 8 - 1, -1):
        sram6.write('{}\t'.format((quant(rnet_act4[i]))))
        sram6.write('{}\t'.format((quant(rnet_bias4[i]))))
    sram6.write('\n')

# rnet-fc2 #############################################################################################################
rnet_fc2 = rnet[12]
rnet_fc2 = np.swapaxes(rnet_fc2, 0, 1)
rnet_fc2 = rnet_fc2.flatten()
rnet_fc2 = rnet_fc2.reshape(16, 16)
for i in range(len(rnet_fc2)):
    for j in range(len(rnet_fc2[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(rnet_fc2[i][j]))))
    sram4.write('\n')

# rnet-fc3 #############################################################################################################
rnet_fc3 = rnet[14]
rnet_fc3 = np.swapaxes(rnet_fc3, 0, 1)
rnet_fc3 = rnet_fc3.flatten()
rnet_fc3 = rnet_fc3.reshape(32, 16)
for i in range(len(rnet_fc3)):
    for j in range(len(rnet_fc3[i]) - 1, -1, -1):
        sram4.write('{}\t'.format((quant(rnet_fc3[i][j]))))
    sram4.write('\n')

# rnet-bias5-bias6 #####################################################################################################
rnet_bias5 = rnet[13]
rnet_bias6 = rnet[15]
for i in range(10):
    sram6.write('{}\t'.format(0))
for i in range(4 - 1, -1, -1):
    sram6.write('{}\t'.format((quant(rnet_bias6[i]))))
for i in range(2 - 1, -1, -1):
    sram6.write('{}\t'.format((quant(rnet_bias5[i]))))
sram6.write('\n')

