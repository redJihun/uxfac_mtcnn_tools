import numpy as np

data = np.load('/home/hong/Documents/python-mtcnn/check_weight/mtcnn_weights.npy', allow_pickle=True).tolist()

change_data = {}
change_data_pnet = []
change_data_rnet = []

change_data_pnet.append(data['pnet'][0][:,:,:,:8])
change_data_pnet.append(data['pnet'][1][:8])
change_data_pnet.append(data['pnet'][2][:,:,:8])
change_data_pnet.append(data['pnet'][3][:,:,:8,:])
change_data_pnet.append(data['pnet'][4])
change_data_pnet.append(data['pnet'][5])
change_data_pnet.append(data['pnet'][6])
change_data_pnet.append(data['pnet'][7])
change_data_pnet.append(data['pnet'][8])
change_data_pnet.append(data['pnet'][9])
change_data_pnet.append(data['pnet'][10])
change_data_pnet.append(data['pnet'][11])
change_data_pnet.append(data['pnet'][12])

change_data_rnet.append(data['rnet'][0])
change_data_rnet.append(data['rnet'][1])
change_data_rnet.append(data['rnet'][2])
change_data_rnet.append(data['rnet'][3])
change_data_rnet.append(data['rnet'][4])
change_data_rnet.append(data['rnet'][5])
col = np.zeros((1,2,48,64))
row = np.zeros((3,1,48,64))
foo = np.append(data['rnet'][6], col, axis=0)
foo = np.append(foo, row, axis=1)
change_data_rnet.append(foo)
change_data_rnet.append(data['rnet'][7])
change_data_rnet.append(data['rnet'][8])
change_data_rnet.append(data['rnet'][9])
change_data_rnet.append(data['rnet'][10])
change_data_rnet.append(data['rnet'][11])
change_data_rnet.append(data['rnet'][12])
change_data_rnet.append(data['rnet'][13])
change_data_rnet.append(data['rnet'][14])
change_data_rnet.append(data['rnet'][15])

change_data['pnet'] = change_data_pnet
change_data['rnet'] = change_data_rnet
np.save('shape_change', change_data)

