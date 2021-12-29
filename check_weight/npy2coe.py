import decimal
import math
import struct

import numpy as np


def binary(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num)
    # print('Packed: %s' % repr(packed))

    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    #
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [c for c in packed]
    # print('Integers: %s' % integers)

    # print('Hex: {0:x} {1:x} {2:x} {3:x}'.format(*integers))

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]
    # print('Binaries: %s' % binaries)

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    # print('Stripped: %s' % stripped_binaries)

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    # print('Padded: %s' % padded)

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return hex(int(''.join(padded), 2)).replace('0x', '')
    # return int(''.join(padded), 2) if int(''.join(padded), 2) < 2147483648 else (~int(''.join(padded), 2))

def fixed(num):
    data = num
    # data = data * (2**28)
    # data = int(math.floor(data))
    return (data)

data = np.load('/home/hong/Documents/python-mtcnn/check_weight/mtcnn_weights.npy', allow_pickle=True).tolist()
change_data_pnet = []
change_data_rnet = []
pnet = data['pnet']
rnet = data['rnet']
sram4 = open('sram4.coe', 'w')
sram5 = open('sram5.coe', 'w')
sram6 = open('sram6.coe', 'w')

# pnet-conv1 ###########################################################################################################
pnet_conv1 = data['pnet'][0][:,:,:,:8]
pnet_conv1 = np.transpose(pnet_conv1, (2, 3, 1, 0))
# pnet_conv1 = np.swapaxes(pnet_conv1, 0, 1)
# pnet_conv1 = np.swapaxes(pnet_conv1, 0, 2)
# pnet_conv1 = np.swapaxes(pnet_conv1, 1, 3)
pnet_conv1 = pnet_conv1.flatten()
while len(pnet_conv1) < 224:
    pnet_conv1 = np.append(pnet_conv1, 0)
pnet_conv1 = pnet_conv1.reshape(14, 16)
for i in range(len(pnet_conv1)):
    for j in range(len(pnet_conv1[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(pnet_conv1[i][j]))))
    sram4.write('\n')

# pnet-bias1-act1 ######################################################################################################
pnet_bias1 = data['pnet'][1]
pnet_act1 = data['pnet'][2].flatten()
# with open('check_weight/sram_6.coe', 'w') as file:
for i in range(4):
    sram6.write('{:>10}\t'.format(fixed(np.float32(pnet_bias1[i]))))
for i in range(4):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act1[i]))))
for i in range(4, 8):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias1[i]))))
for i in range(4, 8):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act1[i]))))
sram6.write('\n')

# pnet-conv2 ###########################################################################################################
pnet_conv2 = data['pnet'][3][:,:,:8,:]
pnet_conv2 = np.transpose(pnet_conv2, (2, 3, 1, 0))
# pnet_conv2 = np.swapaxes(pnet_conv2, 0, 1)
# pnet_conv2 = np.swapaxes(pnet_conv2, 0, 2)
# pnet_conv2 = np.swapaxes(pnet_conv2, 1, 3)
pnet_conv2 = pnet_conv2.flatten()
while len(pnet_conv2) < 1152:
    pnet_conv2 = np.append(pnet_conv2, 0)
pnet_conv2 = pnet_conv2.reshape(72, 16)
# with open('check_weight/sram_4.coe', 'w') as file:
for i in range(len(pnet_conv2)):
    for j in range(len(pnet_conv2[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(pnet_conv2[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/pnet_conv2.coe', pnet_conv2, fmt='%f', delimiter='\t')

# pnet-bias2-act2 ######################################################################################################
pnet_bias2 = data['pnet'][4]
pnet_act2 = data['pnet'][5].flatten()
# with open('check_weight/pnet_bias2_act2.coe', 'w') as file:
# with open('check_weight/sram_6.coe', 'w') as file:
for i in range(4):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias2[i]))))
for i in range(4):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act2[i]))))
for i in range(4, 8):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias2[i]))))
for i in range(4, 8):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act2[i]))))
sram6.write('\n')
for i in range(8, 12):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias2[i]))))
for i in range(8, 12):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act2[i]))))
for i in range(12, 16):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias2[i]))))
for i in range(12, 16):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act2[i]))))
sram6.write('\n')

# pnet-conv3 ###########################################################################################################
pnet_conv3_1 = data['pnet'][6][:,:,:,:16]
pnet_conv3_2 = data['pnet'][6][:,:,:,16:]
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
    for j in range(len(pnet_conv3[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(pnet_conv3[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/pnet_conv3.coe', pnet_conv3, fmt='%f', delimiter='\t')

# pnet-bias3-act3 ######################################################################################################
pnet_bias3 = data['pnet'][7]
pnet_act3 = data['pnet'][8].flatten()
# with open('check_weight/pnet_bias3_act3.coe', 'w') as file:
for index in range(4):
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias3[i]))))
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act3[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias3[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_act3[i]))))
    sram6.write('\n')

# pnet-conv4 ###########################################################################################################
pnet_conv4 = data['pnet'][9]
pnet_conv4 = np.transpose(pnet_conv4, (3, 2, 0, 1))
# pnet_conv4 = np.swapaxes(pnet_conv4, 2, 3)
# pnet_conv4 = np.swapaxes(pnet_conv4, 0, 2)
# pnet_conv4 = np.swapaxes(pnet_conv4, 1, 3)
pnet_conv4 = pnet_conv4.flatten()
while len(pnet_conv4) < 64:
    pnet_conv4 = np.append(pnet_conv4, 0)
pnet_conv4 = pnet_conv4.reshape(4, 16)
for i in range(len(pnet_conv4)):
    for j in range(len(pnet_conv4[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(pnet_conv4[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/pnet_conv4.coe', pnet_conv4, fmt='%f', delimiter='\t')

# pnet-conv5 ###########################################################################################################
pnet_conv5 = data['pnet'][11]
pnet_conv5 = np.transpose(pnet_conv5, (3, 2, 0, 1))
# pnet_conv5 = np.swapaxes(pnet_conv5, 2, 3)
# pnet_conv5 = np.swapaxes(pnet_conv5, 0, 2)
# pnet_conv5 = np.swapaxes(pnet_conv5, 1, 3)
pnet_conv5 = pnet_conv5.flatten()
while len(pnet_conv5) < 128:
    pnet_conv5 = np.append(pnet_conv5, 0)
pnet_conv5 = pnet_conv5.reshape(8, 16)
for i in range(len(pnet_conv5)):
    for j in range(len(pnet_conv5[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(pnet_conv5[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/pnet_conv5.coe', pnet_conv5, fmt='%f', delimiter='\t')

# pnet-bias4-bias5 ######################################################################################################
pnet_bias4 = data['pnet'][10]
pnet_bias5 = data['pnet'][12]
# with open('check_weight/pnet_bias4_bias5.coe', 'w') as file:
for i in range(2):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias4[i]))))
for i in range(4):
    sram6.write('{:>10}\t' .format (fixed(np.float32(pnet_bias5[i]))))
for i in range(10):
    sram6.write('{:>10}\t'.format(0))
sram6.write('\n')

# rnet-conv1 ###########################################################################################################
rnet_conv1_1 = data['rnet'][0][:,:,:,:16]
rnet_conv1_2 = data['rnet'][0][:,:,:,16:]
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
    for j in range(len(rnet_conv1[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(rnet_conv1[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/rnet_conv1.coe', rnet_conv1, fmt='%f', delimiter='\t')

# rnet-bias1-act1 ######################################################################################################
rnet_bias1 = data['rnet'][1]
rnet_act1 = data['rnet'][2].flatten()
# with open('check_weight/rnet_bias1_act1.coe', 'w') as file:
for index in range(3):
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias1[i]))))
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act1[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias1[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act1[i]))))
    sram6.write('\n')
for i in range(20, 24):
    sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias1[i]))))
for i in range(20, 24):
    sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act1[i]))))
for i in range(8):
    sram6.write('{:>10}\t' .format( 0))
sram6.write('\n')

# rnet-conv2 ###########################################################################################################
rnet_conv2_1 = data['rnet'][3][:,:,:,:16]
rnet_conv2_2 = data['rnet'][3][:,:,:,16:32]
rnet_conv2_3 = data['rnet'][3][:,:,:,32:]
rnet_conv2_1 = np.transpose(rnet_conv2_1, (2,3,1,0))
rnet_conv2_1 = rnet_conv2_1.flatten()
rnet_conv2_2 = np.transpose(rnet_conv2_2, (2,3,1,0))
rnet_conv2_2 = rnet_conv2_2.flatten()
rnet_conv2_3 = np.transpose(rnet_conv2_3, (2,3,1,0))
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
    for j in range(len(rnet_conv2[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(rnet_conv2[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/rnet_conv2.coe', rnet_conv2, fmt='%f', delimiter='\t')

# rnet-bias2-act2 ######################################################################################################
rnet_bias2 = data['rnet'][4]
rnet_act2 = data['rnet'][5].flatten()
# with open('check_weight/rnet_bias2_act2.coe', 'w') as file:
for index in range(6):
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias2[i]))))
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act2[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias2[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act2[i]))))
    sram6.write('\n')

# rnet-conv3 ###########################################################################################################
rnet_conv3_1 = data['rnet'][6][:,:,:,:16]
rnet_conv3_2 = data['rnet'][6][:,:,:,16:32]
rnet_conv3_3 = data['rnet'][6][:,:,:,32:48]
rnet_conv3_4 = data['rnet'][6][:,:,:,48:]
col = np.zeros((48, 16, 2, 1))
row = np.zeros((48, 16, 1, 3))
rnet_conv3_1 = np.transpose(rnet_conv3_1, (2,3,1,0))
rnet_conv3_1 = np.append(rnet_conv3_1, col, axis=3)
rnet_conv3_1 = np.append(rnet_conv3_1, row, axis=2)
rnet_conv3_1 = rnet_conv3_1.flatten()
rnet_conv3_2 = np.transpose(rnet_conv3_2, (2,3,1,0))
rnet_conv3_2 = np.append(rnet_conv3_2, col, axis=3)
rnet_conv3_2 = np.append(rnet_conv3_2, row, axis=2)
rnet_conv3_2 = rnet_conv3_2.flatten()
rnet_conv3_3 = np.transpose(rnet_conv3_3, (2,3,1,0))
rnet_conv3_3 = np.append(rnet_conv3_3, col, axis=3)
rnet_conv3_3 = np.append(rnet_conv3_3, row, axis=2)
rnet_conv3_3 = rnet_conv3_3.flatten()
rnet_conv3_4 = np.transpose(rnet_conv3_4, (2,3,1,0))
rnet_conv3_4 = np.append(rnet_conv3_4, col, axis=3)
rnet_conv3_4 = np.append(rnet_conv3_4, row, axis=2)
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
    for j in range(len(rnet_conv3[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(rnet_conv3[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/rnet_conv3.coe', rnet_conv3, fmt='%f', delimiter='\t')

# rnet-bias3-act3 ######################################################################################################
rnet_bias3 = data['rnet'][7]
rnet_act3 = data['rnet'][8].flatten()
# with open('check_weight/rnet_bias3_act3.coe', 'w') as file:
for index in range(8):
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias3[i]))))
    for i in range(index*8, index*8+4):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act3[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias3[i]))))
    for i in range(index*8+4, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act3[i]))))
    sram6.write('\n')

# rnet-fc1 #############################################################################################################
rnet_fc1 = data['rnet'][9]
rnet_fc1 = np.swapaxes(rnet_fc1, 0, 1)
rnet_fc1 = rnet_fc1.flatten()
rnet_fc1 = rnet_fc1.reshape(4608, 16)
for i in range(len(rnet_fc1)):
    for j in range(len(rnet_fc1[i])):
        sram5.write('{:>10}\t'.format(fixed(np.float32(rnet_fc1[i][j]))))
    sram5.write('\n')
# np.savetxt('check_weight/rnet_fc1.coe', rnet_fc1, fmt='%f', delimiter='\t')

# rnet-bias4-act4 ######################################################################################################
rnet_bias4 = data['rnet'][10]
rnet_act4 = data['rnet'][11].flatten()
# with open('check_weight/rnet_bias4_act4.coe', 'w') as file:
for index in range(16):
    for i in range(index*8, index*8+8):
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias4[i]))))
        sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_act4[i]))))
    sram6.write('\n')

# rnet-fc2 #############################################################################################################
rnet_fc2 = data['rnet'][12]
rnet_fc2 = np.swapaxes(rnet_fc2, 0, 1)
rnet_fc2 = rnet_fc2.flatten()
rnet_fc2 = rnet_fc2.reshape(16, 16)
for i in range(len(rnet_fc2)):
    for j in range(len(rnet_fc2[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(np.float32(rnet_fc2[i][j])))))
    sram4.write('\n')
# np.savetxt('check_weight/rnet_fc2.coe', rnet_fc2, fmt='%f', delimiter='\t')

# rnet-fc3 #############################################################################################################
rnet_fc3 = data['rnet'][14]
rnet_fc3 = np.swapaxes(rnet_fc3, 0, 1)
rnet_fc3 = rnet_fc3.flatten()
rnet_fc3 = rnet_fc3.reshape(32, 16)
for i in range(len(rnet_fc3)):
    for j in range(len(rnet_fc3[i])):
        sram4.write('{:>10}\t'.format(fixed(np.float32(rnet_fc3[i][j]))))
    sram4.write('\n')
# np.savetxt('check_weight/rnet_fc3.coe', rnet_fc3, fmt='%f', delimiter='\t')

# rnet-bias5-bias6 #####################################################################################################
rnet_bias5 = data['rnet'][13]
rnet_bias6 = data['rnet'][15]
# with open('check_weight/rnet_bias5_bias6.coe', 'w') as file:
for i in range(2):
    sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias5[i]))))
for i in range(4):
    sram6.write('{:>10}\t' .format (fixed(np.float32(rnet_bias6[i]))))
for i in range(10):
    sram6.write('{:>10}\t'.format(0))
sram6.write('\n')
# np.flatten(data['pnet'])


for item in pnet:
    print(np.shape(item))
for item in rnet:
    print(np.shape(item))


# print(struct.unpack('<H', np.float32(rnet_bias5[0]).tobytes()[0]))
# print('{} : {}'.format(rnet_bias5[0], bytes(rnet_bias5[0]).hex()))
# print('{} : {}'.format(rnet_bias5[1], bytes(rnet_bias5[1]).hex()))
# print('{} : {}'.format(rnet_bias6[0], bytes(rnet_bias6[0]).hex()))
# print('{} : {}'.format(rnet_bias6[1], bytes(rnet_bias6[1]).hex()))
# print('{} : {}'.format(rnet_bias6[2], bytes(rnet_bias6[2]).hex()))
# print('{} : {}'.format(rnet_bias6[3], bytes(rnet_bias6[3]).hex()))
# print('{}\t=>\t{}'.format(pnet_conv1[0][5], np.float32(pnet_conv1[0][5])))
# print('{}\t=>\t{}'.format(bytes(pnet_conv1[0][5]).hex(), bytes(np.float32(pnet_conv1[0][5])).hex()))
# print('64: {}'.format(decimal.Decimal(pnet_conv1[0][0]).as_tuple()))
# print('64: {}'.format(math.frexp(pnet_conv1[0][0])))
# print(fixed(np.float32(pnet_conv1[0][5])))
# for item in struct.pack('!f', pnet_conv1[0][0]):
#     print(item)
# print('{:0>8b}'.format(c) for c in struct.pack('!f', pnet_conv1[0][0]))
# foo = np.float32(pnet_conv1[0][0])
# print(type(foo))
# print('32: {}'.format(decimal.Decimal(foo).as_tuple()))

