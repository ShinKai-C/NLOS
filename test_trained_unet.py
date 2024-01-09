import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import torchvision.transforms.functional as F
import torch
import torch.utils.data as Data
from torch import nn
import os
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from func import *
from models import UNET


h_tr, w_tr, h_dep, w_dep = 32, 32, 128, 128
time_bin = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNET(h_tr, w_tr, time_bin).to(device)
net.load_state_dict(torch.load("trained_model/unet/unet_params_retrain_epoch_4.pth", map_location=device))
root_path = 'E:\ShapeNet_32'
# root_path = 'D:\\Science_Research\\papers\\learned_feature_embeddings_for_NLOS\\NLOSFeatureEmbeddings-main\\ShapeNetRenderer\\data\\bunny-model'

all_files = get_filelist(root_path, [], '_32.mat')
print(len(all_files))

BATCH_SIZE = 5
NUM_OF_TEST_DATA = 6000
# NUM_OF_TRAINING_DATA =
NUM_OF_TRAINING_DATA = len(all_files) - NUM_OF_TEST_DATA - 1

net_rand_list = np.load('trained_model/unet/random_sequence_unet_retrain.npy')


#########################################################################################################################

'''Draw random-selected results'''


# # # 取测试集样本
# # random.seed(2)
# # random_num = random_sample(len(net_rand_list) - NUM_OF_TEST_DATA, len(net_rand_list), BATCH_SIZE)
#
# # 取训练集样本
# random.seed(1)
# random_num = random_sample(0, len(net_rand_list) - NUM_OF_TEST_DATA, BATCH_SIZE)
#
# test_x = torch.zeros((BATCH_SIZE, time_bin, h_tr, w_tr), dtype=torch.float32)
# test_y = torch.zeros((BATCH_SIZE, h_dep, w_dep), dtype=torch.float32)
#
# for i in range(BATCH_SIZE):
#
#     test_x[i, :, :, :] = get_data(all_files[net_rand_list[random_num[i]]], 512)
#
#     folder = os.path.dirname(all_files[net_rand_list[random_num[i]]])
#
#     depth_img = get_filelist(folder, [], '.png')
#
#     test_y[i, :, :] =get_label(depth_img[0])
#
#
#     # test_x[i, :, :, :] = get_data(all_files[0], 512)
#     #
#     # folder = os.path.dirname(all_files[0])
#     #
#     # depth_img = get_filelist(folder, [], '.png')
#     #
#     # test_y[i, :, :] =get_label(depth_img[0])
#
#
# f, a = plt.subplots(2, BATCH_SIZE, figsize=(50, 20))
# plt.ion()   # continuously plot
#
#
# loss_func = nn.MSELoss()
#
# t_x = test_x.to(device)
# t_y = test_y.view((-1, h_dep * w_dep)).to(device)
# t_out = net(t_x)
#
#
# for j in range(t_x.shape[0]):
#     # original data (first row) for viewing
#     a[0][j].clear()
#     gt = np.reshape(t_y.data.cpu().numpy()[j], (h_dep, w_dep))
#     a[0][j].imshow(gt, cmap='gray')
#     a[0][j].set_xticks(())
#     a[0][j].set_yticks(())
#     # test output (second row) for viewing
#     a[1][j].clear()
#     rc = np.reshape(t_out.data.cpu().numpy()[j], (h_dep, w_dep))
#     a[1][j].imshow(rc, cmap='gray')  # Variable 转 numpy需要加.data，可见2.2节
#     a[1][j].set_xticks(())
#     a[1][j].set_yticks(())
#     psnr = peak_signal_noise_ratio(gt, rc, data_range=1)
#     ssim = structural_similarity(gt, rc, data_range=1)
#     mse = mean_squared_error(gt, rc)
#     a[1][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (psnr, ssim, mse), fontsize=20)
#
# plt.draw();plt.pause(0.05)
#
# plt.ioff()
# plt.show()

#########################################################################################################################

'''Draw Loss'''

# train_loss = np.load('loss/unet/unet_train_loss.npy')
# test_loss = np.load('loss/unet/unet_test_loss.npy')
#
# plt.ion()
#
# for i in range(train_loss.shape[0]):
#     plt.plot(range(0, train_loss.shape[1]*100, 100), train_loss[i], label='epoch%d'%i)
#     plt.xlabel('step')
#     plt.ylabel('loss')
#     plt.ylim((0, 0.05))
#     plt.title('train_loss')
#     plt.legend()
#
# # for i in range(test_loss.shape[0]):
# #     plt.plot(range(0, test_loss.shape[1]*100, 100), test_loss[i], label='epoch%d'%i)
# #     plt.xlabel('step')
# #     plt.ylabel('loss')
# #     plt.title('test_loss')
# #     plt.legend()
#
#
# plt.ioff()
# plt.show()


#########################################################################################################################

'''Show PSNR/SSIM/MSE of the whole train/test dataset'''
# from tqdm import tqdm
# from MyDataset import Dataset_transient_label_pair
#
#
# my_train_dataset = Dataset_transient_label_pair(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, net_rand_list,
#                  time_bin, h_tr, w_tr, h_dep, w_dep, flag=0, need_norm=1)
#
# my_test_dataset = Dataset_transient_label_pair(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, net_rand_list,
#                  time_bin, h_tr, w_tr, h_dep, w_dep, flag=1, need_norm=1)
#
# train_loader = Data.DataLoader(
#     dataset=my_train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True
# )
# test_loader = Data.DataLoader(
#     dataset=my_test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True
# )
#
# psnr_sum, ssim_sum, mse_sum = 0, 0, 0
# # 训练集
# num_of_data = len(my_train_dataset) // BATCH_SIZE * BATCH_SIZE
# # # 测试集
# # num_of_data = len(my_test_dataset) // BATCH_SIZE * BATCH_SIZE
#
# with tqdm(total=num_of_data//BATCH_SIZE, ncols=50) as pbar:
#
#     # 训练集
#     for _, (trans, y) in enumerate(train_loader):
#     # # 测试集
#     # for _, (trans, y) in enumerate(test_loader):
#
#         pbar.update(1)
#
#         b_y = y.to(device)
#         b_trans = trans.to(device)
#
#         out = net(b_trans)
#
#         for k in range(b_trans.shape[0]):
#
#             gt = np.reshape(b_y.data.cpu().numpy()[k], (h_dep, w_dep))
#             rc = np.reshape(out.data.cpu().numpy()[k], (h_dep, w_dep))
#
#             psnr = peak_signal_noise_ratio(gt, rc, data_range=1)
#             ssim = structural_similarity(gt, rc, data_range=1)
#             mse = mean_squared_error(gt, rc)
#
#             psnr_sum = psnr_sum + psnr
#             ssim_sum = ssim_sum + ssim
#             mse_sum = mse_sum + mse
#
#
# mean_psnr, mean_ssim, mean_mse = psnr_sum/num_of_data, ssim_sum/num_of_data, mse_sum/num_of_data
# print('PSNR, SSIM, MSE of the whole dataset are %f, %f, %f' %(mean_psnr, mean_ssim, mean_mse))