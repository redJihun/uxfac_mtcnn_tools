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
    return (int(''.join(padded), 2))
    # return int(''.join(padded), 2) if int(''.join(padded), 2) < 2147483648 else (~int(''.join(padded), 2))


def tohex(val, nbits):
  return hex((val + (1 << nbits)) % (1 << nbits)).replace('0x', '')


# data = np.load('p_oc_8.npy', allow_pickle=True).tolist()
data = np.load('/home/hong/Documents/python-mtcnn/mtcnn_weight_coe2npy/converted_npy/12_06_162622_weight.npy', allow_pickle=True).tolist()
pnet = data['pnet']
rnet = data['rnet']

sram4 = open('outputs/sram4.coe', 'w')
sram5 = open('outputs/sram5.coe', 'w')
sram6 = open('outputs/sram6.coe', 'w')

########################################################################################################################
# pnet-conv1
pnet_conv1 = pnet[0]
print(np.shape(pnet_conv1))
pnet_conv1 = pnet_conv1.flatten()
while len(pnet_conv1) < 224:
    pnet_conv1 = np.append(pnet_conv1, 0)
pnet_conv1 = pnet_conv1.reshape(14, 16)
for i in range(len(pnet_conv1)):
    for j in range(len(pnet_conv1[i])-1, -1, -1):
        sram4.write('{:02x}'.format((int(tohex(pnet_conv1[i][j], 8), 16))))
    sram4.write('\n')

########################################################################################################################
# pnet-bias1-act1
pnet_bias1 = np.zeros(pnet[1].shape)
pnet_act1 = pnet[2]
bi, ai = 7, 7
for i in range(2):
    for j in range(4):
        sram6.write('{:02x}'.format((int(tohex(pnet_act1[ai], 8), 16))))
        ai -= 1
    for j in range(4):
        # sram6.write('{:02x}'.format(int(tohex(pnet_bias1[bi], 8), 16)))
        sram6.write('{:02d}'.format(int(pnet_bias1[bi])))
        bi -= 1
sram6.write('\n')

########################################################################################################################
# pnet-conv2
pnet_conv2 = pnet[3]
pnet_conv2 = pnet_conv2.flatten()
while len(pnet_conv2) < 1152:
    pnet_conv2 = np.append(pnet_conv2, 0)
pnet_conv2 = pnet_conv2.reshape(72, 16)
for i in range(len(pnet_conv2)):
    for j in range(len(pnet_conv2[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(pnet_conv2[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# pnet-bias2-act2
pnet_bias2 = np.zeros(pnet[4].shape)
pnet_act2 = pnet[5]
for index in range(2):
    for i in range(index*8+8-1, index*8+4-1, -1):
        print(i)
        sram6.write('{:02x}' .format(int(tohex(pnet_act2[i], 8), 16)))
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02d}'.format(int(pnet_bias2[i])))
    for i in range(index*8+4-1, index*8-1, -1):
        print(i)
        sram6.write('{:02x}' .format(int(tohex(pnet_act2[i], 8), 16)))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02d}'.format(int(pnet_bias2[i])))
    sram6.write('\n')

########################################################################################################################
# pnet-conv3
pnet_conv3_1 = pnet[6][:,:16,:,:]
pnet_conv3_2 = pnet[6][:,16:,:,:]
pnet_conv3_1 = pnet_conv3_1.flatten()
pnet_conv3_2 = pnet_conv3_2.flatten()
while len(pnet_conv3_1) < 2304:
    pnet_conv3_1 = np.append(pnet_conv3_1, 0)
while len(pnet_conv3_2) < 2304:
    pnet_conv3_2 = np.append(pnet_conv3_2, 0)
pnet_conv3 = np.append(pnet_conv3_1, pnet_conv3_2)
pnet_conv3 = pnet_conv3.reshape(288, 16)
for i in range(len(pnet_conv3)):
    for j in range(len(pnet_conv3[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(pnet_conv3[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# pnet-bias3-act3
pnet_bias3 = np.zeros(pnet[7].shape)
pnet_act3 = pnet[8]
for index in range(4):
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02x}' .format(int(tohex(pnet_act3[i], 8), 16)))
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02d}'.format(int(pnet_bias3[i])))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02x}' .format(int(tohex(pnet_act3[i], 8), 16)))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02d}'.format(int(pnet_bias3[i])))
    sram6.write('\n')

########################################################################################################################
# pnet-conv4
pnet_conv4 = pnet[11]
pnet_conv4 = np.swapaxes(pnet_conv4, 0, 1)
pnet_conv4 = pnet_conv4.flatten()
pnet_conv4 = pnet_conv4.reshape(4, 16)
for i in range(len(pnet_conv4)):
    for j in range(len(pnet_conv4[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(pnet_conv4[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# pnet-conv5
pnet_conv5 = pnet[9]
pnet_conv5 = np.swapaxes(pnet_conv5, 0, 1)
pnet_conv5 = pnet_conv5.flatten()
pnet_conv5 = pnet_conv5.reshape(8, 16)
for i in range(len(pnet_conv5)):
    for j in range(len(pnet_conv5[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(pnet_conv5[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# pnet-bias4-bias5
pnet_bias4 = pnet[12]
pnet_bias5 = pnet[10]
for i in range(32):
    sram6.write('0')
sram6.write('\n')

########################################################################################################################
# rnet-conv1
rnet_conv1_1 = rnet[0][:,:16,:,:]
rnet_conv1_1 = rnet_conv1_1.flatten()
rnet_conv1_2 = rnet[0][:,16:,:,:]
rnet_conv1_2 = rnet_conv1_2.flatten()
while len(rnet_conv1_2) < 352:
    rnet_conv1_2 = np.append(rnet_conv1_2, 0)
rnet_conv1 = np.append(rnet_conv1_1, rnet_conv1_2)
rnet_conv1 = rnet_conv1.reshape(49, 16)
for i in range(len(rnet_conv1)):
    for j in range(len(rnet_conv1[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(rnet_conv1[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# rnet-bias1-act1
rnet_bias1 = np.zeros(rnet[1].shape)
rnet_act1 = rnet[2]
for index in range(3):
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act1[i], 8), 16)))
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02d}'.format(int(rnet_bias1[i])))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act1[i], 8), 16)))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02d}'.format(int(rnet_bias1[i])))
    sram6.write('\n')
for i in range(16):
    sram6.write('0')
for i in range(24-1, 20-1, -1):
    sram6.write('{:02x}' .format(int(tohex(rnet_act1[i], 8), 16)))
for i in range(24 - 1, 20 - 1, -1):
    sram6.write('{:02d}'.format(int(rnet_bias1[i])))
sram6.write('\n')

########################################################################################################################
# rnet-conv2
rnet_conv2_1 = rnet[3][:,:16,:,:]
rnet_conv2_2 = rnet[3][:,16:32,:,:]
rnet_conv2_3 = rnet[3][:,32:,:,:]
rnet_conv2_1 = rnet_conv2_1.flatten()
rnet_conv2_2 = rnet_conv2_2.flatten()
rnet_conv2_3 = rnet_conv2_3.flatten()
rnet_conv2 = np.append(rnet_conv2_1, rnet_conv2_2)
rnet_conv2 = np.append(rnet_conv2, rnet_conv2_3)
rnet_conv2 = rnet_conv2.reshape(756, 16)
for i in range(len(rnet_conv2)):
    for j in range(len(rnet_conv2[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(rnet_conv2[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# rnet-bias2-act2
rnet_bias2 = np.zeros(rnet[4].shape)
rnet_act2 = rnet[5]
for index in range(6):
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act2[i], 8), 16)))
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02d}'.format(int(rnet_bias2[i])))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act2[i], 8), 16)))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02d}'.format(int(rnet_bias2[i])))
    sram6.write('\n')

########################################################################################################################
# rnet-conv3
rnet_conv3_1 = rnet[6][:,:16,:,:]
rnet_conv3_2 = rnet[6][:,16:32,:,:]
rnet_conv3_3 = rnet[6][:,32:48,:,:]
rnet_conv3_4 = rnet[6][:,48:,:,:]
rnet_conv3_1 = rnet_conv3_1.flatten()
rnet_conv3_2 = rnet_conv3_2.flatten()
rnet_conv3_3 = rnet_conv3_3.flatten()
rnet_conv3_4 = rnet_conv3_4.flatten()
rnet_conv3 = np.append(rnet_conv3_1, rnet_conv3_2)
rnet_conv3 = np.append(rnet_conv3, rnet_conv3_3)
rnet_conv3 = np.append(rnet_conv3, rnet_conv3_4)
rnet_conv3 = rnet_conv3.reshape(1728, 16)
for i in range(len(rnet_conv3)):
    for j in range(len(rnet_conv3[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(rnet_conv3[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# rnet-bias3-act3
rnet_bias3 = np.zeros(rnet[7].shape)
rnet_act3 = rnet[8]
for index in range(8):
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act3[i], 8), 16)))
    for i in range(index*8+8-1, index*8+4-1, -1):
        sram6.write('{:02d}'.format(int(rnet_bias3[i])))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act3[i], 8), 16)))
    for i in range(index*8+4-1, index*8-1, -1):
        sram6.write('{:02d}'.format(int(rnet_bias3[i])))
    sram6.write('\n')

########################################################################################################################
# rnet-fc1
rnet_fc1 = rnet[9]
rnet_fc1 = rnet_fc1.flatten()
rnet_fc1 = rnet_fc1.reshape(4608, 16)
for i in range(len(rnet_fc1)):
    for j in range(len(rnet_fc1[i])-1, -1, -1):
        sram5.write('{:02x}'.format(int(tohex(rnet_fc1[i][j], 8), 16)))
    sram5.write('\n')

########################################################################################################################
# rnet-bias4-act4
rnet_bias4 = np.zeros(rnet[10].shape)
rnet_act4 = rnet[11]
for index in range(16):
    for i in range(index*8+8-1, index*8-1, -1):
        sram6.write('{:02x}' .format(int(tohex(rnet_act4[i], 8), 16)))
        sram6.write('{:02d}'.format(int(rnet_bias4[i])))
    sram6.write('\n')

########################################################################################################################
# rnet-fc2
rnet_fc2 = rnet[14]
rnet_fc2 = rnet_fc2.flatten()
rnet_fc2 = rnet_fc2.reshape(16, 16)
for i in range(len(rnet_fc2)):
    for j in range(len(rnet_fc2[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(rnet_fc2[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# rnet-fc3
rnet_fc3 = rnet[12]
rnet_fc3 = rnet_fc3.flatten()
rnet_fc3 = rnet_fc3.reshape(32, 16)
for i in range(len(rnet_fc3)):
    for j in range(len(rnet_fc3[i])-1, -1, -1):
        sram4.write('{:02x}'.format(int(tohex(rnet_fc3[i][j], 8), 16)))
    sram4.write('\n')

########################################################################################################################
# rnet-bias5-bias6
rnet_bias5 = np.zeros(rnet[15].shape)
rnet_bias6 = np.zeros(rnet[13].shape)
for i in range(32):
    sram6.write('0')
