import torch
from torch.nn import init
import scipy.io as sio  # 用于读取.mat文件
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import os
import random
import numpy as np
import math
from einops import rearrange


def get_filelist(path, all_files, suffix=''):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            get_filelist(cur_path, all_files, suffix)
        else:
            if ((suffix != '' and file.endswith(suffix)) or suffix == ''):
                all_files.append(cur_path)

    return all_files

def random_sample(bigin, end, NUM):

    items = range(bigin, end)
    rand_list = random.sample(items, NUM)

    return rand_list

#  是UNet的输入，也是CSNet的标签
def get_data(tran_loc, time_bin):

    transient = sio.loadmat(tran_loc)
    transient_data = transient['photon_counts']
    transient_data = transient_data[:, :, :time_bin]
    transient_data = transient_data.transpose([2, 0, 1])
    # transient_data = normalization(transient_data)  # 改在主函数里归一化了
    transient_data = torch.from_numpy(transient_data).to(torch.float32)  # (time_bin, h, w)
    return transient_data

def get_data_32(tran_loc, time_bin):

    transient = sio.loadmat(tran_loc)
    transient_data = transient['photon_counts']
    transient_data = transient_data[::2, ::2, :time_bin]
    transient_data = transient_data.transpose([2, 0, 1])
    # transient_data = normalization(transient_data)  # 改在主函数里归一化了
    transient_data = torch.from_numpy(transient_data).to(torch.float32)  # (time_bin, h, w)
    return transient_data

def get_label(dep_loc):

    img = mpimg.imread(dep_loc)
    depth_data = img[64:192, 64:192, 0]  # 灰度图三通道值相同，0-1之间的值
    depth_data = torch.from_numpy(depth_data).to(torch.float32)

    return depth_data

def normalization(data):

    minVals = torch.min(data)
    maxVals = torch.max(data)
    if not minVals == maxVals:
        ranges = maxVals - minVals
        normData = (data - minVals)/ranges
    else:
        normData = data

    return normData

def get_compressed(mask, transient_data):

    # transient_data = transient_data.reshape((transient_data.shape[0],
    #                                          transient_data.shape[1] * transient_data.shape[2])).T
    transient_data = rearrange(transient_data, 'c h w -> (h w) c')
    compressed_data = mask.mm(transient_data)

    return compressed_data


def get_mask(cr):
    if cr in [1, 5, 10, 20, 50, 100]:

        if os.path.exists('sensing_matrix/cr%d_32.mat' % cr):

            mask = sio.loadmat('sensing_matrix/cr%d_32.mat' % cr)
            mask = mask['sensing_matrix'].transpose([2, 0, 1])
            # print(mask.dtype)
            mask = torch.from_numpy(mask).to(torch.float)
            # mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))
            mask = rearrange(mask, 'c h w -> c (h w)')

        # if os.path.exists('sensing_matrix/cr%d_32_random_gaussian.mat' % cr):
        #
        #     mask = sio.loadmat('sensing_matrix/cr%d_32_random_gaussian.mat' % cr)
        #     mask = mask['sensing_matrix']
        #     mask = torch.from_numpy(mask).to(torch.float)

        else:
            mask = sio.loadmat('sensing_matrix/cr50_32.mat')
            mask = mask['sensing_matrix']
            # print(mask.dtype)

            meas_num = math.ceil(cr/50 * mask.shape[2])
            samp_list = random_sample(0, mask.shape[2], meas_num)
            sio.savemat('sensing_matrix/cr%d_32.mat' % cr,
                        {'sensing_matrix': mask[:, :, samp_list]})
            print('save mat done!')

            mask = torch.from_numpy(mask.transpose([2, 0, 1])).to(torch.float)
            # mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))
            mask = rearrange(mask, 'c h w -> c (h w)')
            mask = mask[samp_list, :]

    else:
        raise Exception("Invalid compress rate!")

    return mask


def transient2frames(path, rand_list, tr_loader=get_data, NUM_OF_TRAINING_DATA=0, time_bin=512):
    transient_dataset, test_tr = [], []

    def save_frame_num(sam_num, transient_data, dataset):

        for j in range(time_bin):
            frame = transient_data[j, :, :].squeeze()
            # dataset.append((sam_num, j))


            if torch.any(frame) != 0:
                dataset.append((sam_num, j))
            else:
                continue

    for i in range(len(rand_list)):

        transient_data = tr_loader(path[rand_list[i]], time_bin)

        if (i < NUM_OF_TRAINING_DATA):

            save_frame_num(rand_list[i], transient_data, transient_dataset)

        else:
            save_frame_num(rand_list[i], transient_data, test_tr)

    return transient_dataset, test_tr


def weight_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('batch') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('relu') != -1:
        init.kaiming_uniform_(m)