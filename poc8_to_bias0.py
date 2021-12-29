import numpy as np

poc8 = np.load('p_oc_8.npy', allow_pickle=True).tolist()

pnet = poc8['pnet']
rnet = poc8['rnet']

pnet[1] = np.zeros(pnet[1].shape)
pnet[4] = np.zeros(pnet[4].shape)
pnet[7] = np.zeros(pnet[7].shape)
pnet[10] = np.zeros(pnet[10].shape)
pnet[12] = np.zeros(pnet[12].shape)

rnet[1] = np.zeros(rnet[1].shape)
rnet[4] = np.zeros(rnet[4].shape)
rnet[7] = np.zeros(rnet[7].shape)
rnet[10] = np.zeros(rnet[10].shape)
rnet[13] = np.zeros(rnet[13].shape)
rnet[15] = np.zeros(rnet[15].shape)

poc8_bias0 = {'pnet': pnet, 'rnet': rnet}
np.save('p_oc_8_bias0', poc8_bias0)

poc8_bias0_test = np.load('p_oc_8_bias0.npy', allow_pickle=True).tolist()
# print(pnet[1])
# print(pnet[4])
# print(pnet[7])
# print(pnet[10])
# print(pnet[12])
# print(rnet[1])
# print(rnet[4])
# print(rnet[7])
# print(rnet[10])
# print(rnet[13])
# print(rnet[15])

