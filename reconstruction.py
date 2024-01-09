#coding=UTF-8

import time
import torch
import numpy as np
import scipy.io as sio  # 用于读取.mat文件
import torch.utils.data as Data
from torch import nn
from UNET_3D_2D_parts import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import os
import random
from func import *
from models import UNET
from MyDataset import Dataset_transient_label_pair
from Train import UNET_train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

start = time.time()

# 预定义参数
h_tr, w_tr = 32, 32
h_dep, w_dep = 128, 128  # 256*256裁剪为128*128
time_bin = 512

BATCH_SIZE = 5
LR = 1e-4
EPOCH = 5
root_path = 'E:/ShapeNet_32'

all_files = get_filelist(root_path, [], '.mat')
print(len(all_files))

NUM_OF_TEST_DATA = 6000
# NUM_OF_TRAINING_DATA =
NUM_OF_TRAINING_DATA = len(all_files) - NUM_OF_TEST_DATA - 1

# rand_list = random_sample(1, len(all_files), len(all_files)-1) # 第0个样本作为备用以防止某些样本损坏使程序崩溃
# np.save('trained_model/unet/random_sequence_unet_retrain.npy', rand_list)

unet_rand_list = np.load('trained_model/unet/random_sequence_unet_retrain.npy')
# print(len(unet_rand_list))
rand_list = random_sample(0, len(unet_rand_list)-6000, NUM_OF_TRAINING_DATA)
rand_list.extend(random_sample(len(unet_rand_list)-6000, len(unet_rand_list), NUM_OF_TEST_DATA))
# print(len(rand_list))
rand_list = unet_rand_list[rand_list]

# transient_dataset = []
# depth_dataset = []
# test_x = []
# test_y = []
#
# for i in range(NUM_OF_TRAINING_DATA):
#
#     transient_dataset.append(all_files[rand_list[i]])
#     folder = os.path.dirname(all_files[rand_list[i]])
#     depth_img = get_filelist(folder, [], '.png')
#
#     depth_dataset.append(depth_img[0])
#
#
# for i in range(NUM_OF_TEST_DATA):
#
#     test_x.append(all_files[rand_list[i+NUM_OF_TRAINING_DATA]])
#
#     folder = os.path.dirname(all_files[rand_list[i+NUM_OF_TRAINING_DATA]])
#     depth_img = get_filelist(folder, [], '.png')
#
#     test_y.append(depth_img[0])
#
#
# class MyDataset(Data.Dataset):
#     def __init__(self, flag, data_loader=get_data, label_loader=get_label, norm=normalization):
#
#         self.data = transient_dataset if flag == 0 else test_x
#         self.label = depth_dataset if flag == 0 else test_y
#         self.data_loader = data_loader
#         self.label_loader = label_loader
#         self.norm = norm
#         self.flag = flag
#
#     def __getitem__(self, index):
#         tr = self.data[index]
#         transient = self.data_loader(tr, time_bin)
#         lb = self.label[index]
#         label = self.label_loader(lb)
#         if torch.sum(transient) != 0 and transient.shape == torch.Size([time_bin, h_tr, w_tr]) \
#                 and torch.sum(label) != 0 and label.shape == torch.Size([h_dep, w_dep]):
#             pass
#
#         else:  # 若瞬态数据或者标签有问题，取第一个数据作为默认值
#
#             # print('No. %d %s data is not useful, take the first data as the default.' % (index, 'training' if self.flag==0 else 'testing'))
#
#             tr = all_files[0]
#             transient = self.data_loader(tr, time_bin)
#             lb = get_filelist(os.path.dirname(tr), [], '.png')[0]
#             label = self.label_loader(lb)
#
#         return self.norm(transient), label
#
#     def __len__(self):
#         return len(self.data)
#
#
# my_train_dataset = MyDataset(flag=0)
# my_test_dataset = MyDataset(flag=1)

rate = 500
'''
因为CSNET学习到的是Y=AX中的测量矩阵A，其中Y是压缩值输入，X是瞬态图输出，为了让网络好收敛，对Y进行归一化操作，
此时相应地也要对压缩值进行同样的操作以保证在不同输入下网络学习的始终是A，即，(Y/Y_max)=A(X/Y_max)，
但Y_max远比X_max大，也即作为标签的(X/Y_max)很小，这样网络极有可能学习到空白值，
因此引入rate衡量Y_max与X_max的大小，在对Y归一化的基础上也对X进行一定的范围限制，便于网络收敛，
此时有(Y/Y_max)=A_1(X/Y_max*rate)，也即A_1 = A/rate
'''

my_train_dataset = Dataset_transient_label_pair(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list,
                 time_bin, h_tr, w_tr, h_dep, w_dep, flag=0, need_norm=1, rate=rate)

my_test_dataset = Dataset_transient_label_pair(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list,
                 time_bin, h_tr, w_tr, h_dep, w_dep, flag=1, need_norm=1, rate=rate)

train_loader = Data.DataLoader(
    dataset=my_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
test_loader = Data.DataLoader(
    dataset=my_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print('dataset and labels are done!')
print(time.time()-start)



if __name__ == '__main__':


    u_net = UNET(h_tr, w_tr, time_bin).to(device)
    # print(net._modules)

    # 加载之前训练好的网络继续训练
    # u_net.load_state_dict(torch.load("trained_model/unet/unet_params_retrain_epoch_1.pth", map_location=device))
    # net.apply(weight_init)  # 权重初始化
    # pytorch中的model.apply(fn)会递归地将函数fn应用到父模块的每个子模块submodule，也包括model这个父模块自身。经常用于初始化init_weights的操作

    optimizer = torch.optim.Adam(u_net.parameters(), lr=LR, weight_decay=1e-4)
    # weight_decay是l2正则化的惩罚因子，要了解l1/l2 loss和l1/l2正则化（惩罚项防止过拟合）的区别
    # pytorch 自带的优化器只有l2正则化
    loss_func = nn.MSELoss()

    #  train
    UNET_train(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, BATCH_SIZE, EPOCH, h_dep, w_dep, device,
               train_loader, test_loader, optimizer, loss_func, u_net)


    # # # initialize figure
    # # f, a = plt.subplots(2, BATCH_SIZE, figsize=(50, 20))
    # # plt.ion()   # continuously plot
    #
    #
    # net_start = time.time()
    # train_loss_full_list, test_loss_full_list = [], []
    # for epoch in range(EPOCH):
    #
    #     train_loss_list, test_loss_list = [], []
    #
    #     for step, (x, y) in enumerate(train_loader):
    #         b_x = x.to(device)
    #         b_y = y.view((-1, h_dep*w_dep)).to(device)
    #
    #         output = net(b_x)
    #         loss = loss_func(output.view((-1, h_dep*w_dep)), b_y)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print('Epoch:', epoch, '|', 'Step:', step, '|', 'train loss:%.6f' % loss.data)
    #
    #         if(step==0): train_loss_list.append(loss.cpu().detach().numpy())
    #
    #         if((step+1)%100==0 or step==(NUM_OF_TRAINING_DATA//BATCH_SIZE)-1):
    #
    #             train_loss_list.append(loss.cpu().detach().numpy())
    #
    #             print('--------------------------------')
    #
    #             net.eval()
    #             with torch.no_grad():
    #
    #                 # t_loss = 0
    #                 sample_seq = random_sample(0, NUM_OF_TEST_DATA//BATCH_SIZE, 1)  # 只有一个元素的list，用来随机选取某个batch展示其深度重建图
    #                 # print(sample_seq[0])
    #                 for seq, (test_data, test_label) in enumerate(test_loader):
    #
    #                     if (seq == sample_seq[0]):
    #
    #                         t_x = test_data.to(device)
    #                         t_y = test_label.to(device)
    #
    #                         t_out = net(t_x)
    #                         loss_tmp = loss_func(t_out.view((-1, h_dep * w_dep)), t_y.view((-1, h_dep * w_dep)))
    #                         # t_loss += loss_tmp.data * t_x.shape[0]
    #
    #                         # # 随机展示测试数据效果
    #                         # for j in range(t_x.shape[0]):
    #                         #     # original data (first row) for viewing
    #                         #     a[0][j].clear()
    #                         #     a[0][j].imshow(np.reshape(t_y.data.cpu().numpy()[j], (h_dep, w_dep)), cmap='gray')
    #                         #     a[0][j].set_xticks(())
    #                         #     a[0][j].set_yticks(())
    #                         #     # test output (second row) for viewing
    #                         #     a[1][j].clear()
    #                         #     a[1][j].imshow(np.reshape(t_out.data.cpu().numpy()[j], (h_dep, w_dep)), cmap='gray')  # Variable 转 numpy需要加.data，可见2.2节
    #                         #     a[1][j].set_xticks(())
    #                         #     a[1][j].set_yticks(())
    #                         #
    #                         # plt.draw();plt.pause(1)
    #
    #
    #                         break
    #
    #                     else:
    #                         continue
    #
    #                 del t_out
    #
    #
    #                 # print('Epoch:', epoch, '| test loss:%.6f' % (t_loss / NUM_OF_TEST_DATA))
    #                 # test_loss_list.append(t_loss.cpu() / NUM_OF_TEST_DATA)
    #
    #                 print('Epoch:', epoch, '| test loss:%.6f' % loss_tmp)
    #                 test_loss_list.append(loss_tmp.cpu())
    #                 print('--------------------------------')
    #
    #
    #             net.train()
    #
    #         del output, loss
    #
    #     torch.save(net.state_dict(),
    #                'D:/Science_Research/神经网络/programs/trained_model/unet/unet_params_retrain_epoch_%d.pth' % epoch)
    #
    #     train_loss_full_list.append(train_loss_list)
    #     test_loss_full_list.append(test_loss_list)
    #
    #
    # train_loss_full_list = np.array(train_loss_full_list)
    # np.save('D:/Science_Research/神经网络/programs/loss/unet/unet_retrain_train_loss.npy', train_loss_full_list)
    #
    # test_loss_full_list = np.array(test_loss_full_list)
    # np.save('D:/Science_Research/神经网络/programs/loss/unet/unet_retrain_test_loss.npy', test_loss_full_list)
    #
    # print('net time: %4f' % (time.time()-net_start))

    # train_loss_full_list = np.load('train_loss.npy').tolist()
    # print(train_loss_full_list)
    # test_loss_full_list = np.load('test_loss.npy').tolist()
    # print(test_loss_full_list)

    #
    # plt.ioff()
    # plt.show()
