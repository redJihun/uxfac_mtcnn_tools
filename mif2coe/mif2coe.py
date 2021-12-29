
def bin2hex(bin_list):
    hex_list = []
    for i in range(len(bin_list)):
        row_list = []
        for num in range(32):
            row_list.append(hex(int(bin_list[i][num*4:num*4+4], 2)).replace('0x','').upper())
            # print(hex(int(bin_list[i][num*4:num*4+4], 2)))
        hex_list.append(row_list)
    return hex_list


def mif_to_bin_list(mif_path):
    mif = open(mif_path)
    bin_list = mif.read().splitlines()
    return bin_list


def hex_list_to_coe(hex_list, coe_path):
    with open(coe_path, 'w') as file:
        for i in range(len(hex_list)):
            for j in range(len(hex_list[i])):
                file.write('{}'.format(hex_list[i][j]))
            file.write('\n')


def mif2coe(mif_path, coe_path):
    mif_bin_list = mif_to_bin_list(mif_path)
    hex_list = bin2hex(mif_bin_list)
    hex_list_to_coe(hex_list, coe_path)


if __name__ == '__main__':
    mif_path = '/home/hong/Documents/python-mtcnn/mif2coe/blk_mem_6.mif'
    coe_path = '/home/hong/Documents/python-mtcnn/mif2coe/test.coe'


