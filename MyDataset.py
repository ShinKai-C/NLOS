#coding=UTF-8

import torch
import numpy as np
import os
import torch.utils.data as Data
from torch import nn
from func import *


class Dataset_3D_CSNET_32(Data.Dataset):

    def get_dataset(self, NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list):

        transient_dataset = []
        depth_dataset = []
        test_tr = []
        test_dep = []

        for i in range(NUM_OF_TRAINING_DATA):
            transient_dataset.append(all_files[rand_list[i]])
            folder = os.path.dirname(all_files[rand_list[i]])
            depth_img = get_filelist(folder, [], '.png')

            depth_dataset.append(depth_img[0])

        for i in range(NUM_OF_TEST_DATA):
            test_tr.append(all_files[rand_list[i + NUM_OF_TRAINING_DATA]])

            folder = os.path.dirname(all_files[rand_list[i + NUM_OF_TRAINING_DATA]])
            depth_img = get_filelist(folder, [], '.png')

            test_dep.append(depth_img[0])

        return transient_dataset, depth_dataset, test_tr, test_dep

    def __init__(self, NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list,
                 cr, device, time_bin, h_tr, w_tr, h_dep, w_dep, flag, rate=500,
                 compress=get_compressed, tr_loader=get_data, label_loader=get_label,
                 norm=normalization, mask=get_mask):

        transient_dataset, depth_dataset, test_tr, test_dep \
            = self.get_dataset(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list)

        self.all_files = all_files
        self.tr = transient_dataset if flag == 0 else test_tr
        self.tr_loader = tr_loader
        self.compress = compress
        self.label = depth_dataset if flag == 0 else test_dep
        self.label_loader = label_loader
        self.norm = norm

        self.cr = cr
        self.device = device
        self.rate = rate
        self.mask = mask(self.cr).to(self.device)
        self.time_bin = time_bin
        self.h_tr, self.w_tr, self.h_dep, self.w_dep = h_tr, w_tr, h_dep, w_dep
        self.flag = flag


    def __getitem__(self, index):
        tr_loc = self.tr[index]
        transient = self.tr_loader(tr_loc, self.time_bin).to(self.device)
        compressed = self.compress(self.mask, transient)
        K = torch.max(compressed)
        lb = self.label[index]
        label = self.label_loader(lb).to(self.device)

        # print(transient.shape, compressed.shape, label.shape)
        # print(torch.sum(transient), torch.sum(compressed), torch.sum(label))

        if torch.sum(transient) != 0 and transient.shape == torch.Size([self.time_bin, self.h_tr, self.w_tr]) \
                and torch.sum(compressed) != 0 and compressed.shape == torch.Size([self.mask.shape[0], self.time_bin])\
                and torch.sum(label) != 0 and label.shape == torch.Size([self.h_dep, self.w_dep]):
            pass

        else:  # 若压缩值或者瞬态数据有问题，取第一个数据作为默认值

            # print('No. %d %s data is not useful, take the first data as the default.' % (index, 'training' if self.flag==0 else 'testing'))

            tr_loc = self.all_files[0]
            transient = self.tr_loader(tr_loc, self.time_bin).to(self.device)
            compressed = self.compress(self.mask, transient)
            K = torch.max(compressed)
            lb = get_filelist(os.path.dirname(tr_loc), [], '.png')[0]
            label = self.label_loader(lb).to(self.device)

            # print(transient.shape, compressed.shape, label.shape)
            # print(torch.sum(transient), torch.sum(compressed), torch.sum(label))

        return compressed/K, transient/K*self.rate, label

    def __len__(self):
        return len(self.tr)



class Dataset_2D_CSNET_32(Data.Dataset):

    def get_dataset(self, NUM_OF_TRAINING_DATA, all_files, rand_list):

        transient_dataset, test_tr = transient2frames(all_files, rand_list, NUM_OF_TRAINING_DATA=NUM_OF_TRAINING_DATA)
        random.shuffle(transient_dataset)
        random.shuffle(test_tr)

        return transient_dataset, test_tr

    def __init__(self, NUM_OF_TRAINING_DATA, all_files, rand_list, cr,
                 device, time_bin, h_tr, w_tr, h_dep, w_dep, flag, rate,
                 norm=normalization, tr_loader=get_data, mask=get_mask):

        transient_dataset, test_tr = self.get_dataset(NUM_OF_TRAINING_DATA, all_files, rand_list)

        self.all_files = all_files
        self.tr = transient_dataset if flag == 0 else test_tr
        self.norm = norm
        self.tr_loader = tr_loader
        self.rate = rate
        self.flag = flag
        self.cr = cr
        self.device = device
        self.mask = mask(self.cr).to(self.device)
        self.time_bin = time_bin
        self.h_tr, self.w_tr, self.h_dep, self.w_dep = h_tr, w_tr, h_dep, w_dep

    def __getitem__(self, index):

        (sam_num, bin) = self.tr[index]
        transient = self.tr_loader(self.all_files[sam_num], self.time_bin)
        transient_frame = transient[bin, :, :].to(self.device)
        with torch.no_grad():
            compressed = torch.matmul(self.mask, transient_frame.flatten())
            K = torch.max(compressed)
            # T = torch.max(transient_frame)
            # print('K : %f, T: %f' %(K.item(), T.item()))

        if transient_frame.shape == torch.Size([self.h_tr, self.w_tr]) and compressed.shape == torch.Size([self.mask.shape[0]]):

            pass

        else:  # 若压缩值或者瞬态数据有问题，取第一个数据作为默认值

            print('No. %d %s data is not useful, take the first data as the default.' % (index, 'training' if self.flag==0 else 'testing'))

            (sam_num, bin) = self.tr[0]
            transient = self.tr_loader(self.all_files[sam_num], self.time_bin)
            transient_frame = transient[bin, :, :].to(self.device)
            with torch.no_grad():
                compressed = torch.matmul(self.mask, transient_frame.flatten())
                K = torch.max(compressed)


        return compressed/K, transient_frame/K*self.rate

    def __len__(self):
        return len(self.tr)



class Dataset_transient_label_pair(Data.Dataset):

    def get_dataset(self, NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list):

        transient_dataset = []
        depth_dataset = []
        test_tr = []
        test_dep = []

        for i in range(NUM_OF_TRAINING_DATA):
            transient_dataset.append(all_files[rand_list[i]])
            folder = os.path.dirname(all_files[rand_list[i]])
            depth_img = get_filelist(folder, [], '.png')

            depth_dataset.append(depth_img[0])

        for i in range(NUM_OF_TEST_DATA):
            test_tr.append(all_files[rand_list[i + NUM_OF_TRAINING_DATA]])

            folder = os.path.dirname(all_files[rand_list[i + NUM_OF_TRAINING_DATA]])
            depth_img = get_filelist(folder, [], '.png')

            test_dep.append(depth_img[0])

        return transient_dataset, depth_dataset, test_tr, test_dep


    def __init__(self, NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list,
                 time_bin, h_tr, w_tr, h_dep, w_dep, flag, need_norm, rate=100,
                 data_loader=get_data, label_loader=get_label, norm=normalization):

        transient_dataset, depth_dataset, test_tr, test_dep \
            = self.get_dataset(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, all_files, rand_list)

        self.all_files = all_files
        self.data = transient_dataset if flag == 0 else test_tr
        self.label = depth_dataset if flag == 0 else test_dep
        self.data_loader = data_loader
        self.label_loader = label_loader
        self.norm = norm

        self.flag = flag
        self.need_norm = need_norm
        self.rate = rate
        self.time_bin = time_bin
        self.h_tr, self.w_tr, self.h_dep, self.w_dep = h_tr, w_tr, h_dep, w_dep

    def __getitem__(self, index):
        tr = self.data[index]
        transient = self.data_loader(tr, self.time_bin)
        K = torch.max(transient)
        lb = self.label[index]
        label = self.label_loader(lb)
        if torch.sum(transient) != 0 and transient.shape == torch.Size([self.time_bin, self.h_tr, self.w_tr]) \
                and torch.sum(label) != 0 and label.shape == torch.Size([self.h_dep, self.w_dep]):
            pass

        else:  # 若瞬态数据或者标签有问题，取第一个数据作为默认值

            # print('No. %d %s data is not useful, take the first data as the default.' % (index, 'training' if self.flag==0 else 'testing'))

            tr = self.all_files[0]
            transient = self.data_loader(tr, self.time_bin)
            K = torch.max(transient)
            lb = get_filelist(os.path.dirname(tr), [], '.png')[0]
            label = self.label_loader(lb)

        if self.need_norm:
            return transient/K, label
        else:
            return transient, label

    def __len__(self):
        return len(self.data)