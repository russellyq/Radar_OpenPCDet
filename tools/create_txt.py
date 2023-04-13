# import os
# import numpy as np
# from os import path

# file_list_train, file_list_val, file_list_test = [],  [], []
# root_dir = '/home/newdisk/yanqiao/dataset/Radar_Thermal/training/label_2/'
# number = 0
# for i in range(3411):
#     idx = "%06d" % (i)
#     file_name = root_dir + "%06d.txt" % (i)
#     if path.exists(file_name):
#         number += 1
#         # if number % 4 == 0:
#         #     file_list_val.append(idx)
#         # else:
#         file_list_train.append(idx)
#         # if (number % 8 ==0) and (not number % 5 == 0):
#         #     file_list_val.append(idx)
# # for dir, root, files in os.walk(root_dir):
# #     for file in files:
# #         x = np.random.uniform(0, 99)
# #         if x <=70:
# #             if x <= 10:
# #                 file_list_train.append(file[:-4])
# #                 file_list_val.append(file[:-4])
# #             else:
# #                 file_list_train.append(file[:-4])
        
# #         else:
# #             file_list_val.append(file[:-4])


        

# with open('train.txt', 'w') as f:
#     for file_name in file_list_train:
#         f.writelines(file_name+'\n')
#     f.close()


# # with open('val.txt', 'w') as f:
# #     for file_name in file_list_val:
# #         f.writelines(file_name+'\n')
# #     f.close()

# # root_dir = '/home/newdisk/yanqiao/dataset/LiDAR_RGB/testing/calib/'

# # for dir, root, files in os.walk(root_dir):
# #     for file in files:
# #         file_list_test.append(file[:-4])


# # with open('test.txt', 'w') as f:
# #     for file_name in file_list_test:
# #         f.writelines(file_name+'\n')
# #     f.close()
# import numpy as np
# file_list_val = []
# f=open('train.txt', 'r')
# x=[]
# for line in f.readlines():
#     line=line.strip()
#     x = np.random.uniform(0, 1499)
#     if x <= 500:
#         file_list_val.append(line)
# print(x)

# with open('val.txt', 'w') as f:
#     for file_name in file_list_val:
#         f.writelines(file_name+'\n')
#     f.close()
def create_txt():
    val = []
    train = []
    
    while True:
        val = []
        train = []
    
        for i in range(1785):
            idx = "%06d" % (i)
            if np.random.uniform(0, 1000000)<220000.57:
                val.append(idx)
            else:
                train.append(idx)
        for i in range(1785,2000):
            idx = "%06d" % (i)
            if np.random.uniform(0, 1000000)<500000:
                val.append(idx)
            else:
                train.append(idx)
        
        if len(train) == 1500 and len(val) == 500:
        
            with open('val.txt', 'w') as f:
                for file_name in val:
                    f.writelines(file_name+'\n')
            f.close()
            
            with open('train.txt', 'w') as f:
                for file_name in train:
                    f.writelines(file_name+'\n')
            f.close()
            
            print('oki')
            break
        else:
            print('not oki: ', len(train), len(val))

create_txt()