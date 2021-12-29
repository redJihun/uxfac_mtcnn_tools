import numpy as np

hex_file = open('rgb_211220.txt', 'r')

temp_list = hex_file.read().splitlines()
results_list = []

for item in temp_list:
    results_list.append(np.uint8(int(item[:2],16)))
    results_list.append(np.uint8(int(item[2:4], 16)))
    results_list.append(np.uint8(int(item[4:6], 16)))

print(results_list)
print(len(results_list))

dec_file = open('rgb_211220_C_input.txt', 'w')

for item in results_list:
    dec_file.write('{}\n'.format(item))

