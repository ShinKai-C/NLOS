import time
import torch
from torch import nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import torch.utils.data as Data
from UNET_3D_2D_parts import *
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import os
import random
from func import *
from models import UNET


def CSNET_32_train(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, BATCH_SIZE, EPOCH,
                   time_bin, h_tr, w_tr, h_dep, w_dep, device,
                   train_loader, test_loader, optimizer, loss_func, cs_net
                   , u_net
                   ):

    # initialize figure
    f, a = plt.subplots(2, BATCH_SIZE, figsize=(50, 50))
    cb0, cb1 = {}, {}
    # cb2, cb3 = {}, {}  '''f, a = plt.subplots(4, BATCH_SIZE, figsize=(10, 10))'''

    plt.ion()  # continuously plot

    net_start = time.time()
    train_loss_full_list, test_loss_full_list = [], []
    for epoch in range(EPOCH):

        train_loss_list, test_loss_list = [], []

        for step, (x, trans, y) in enumerate(train_loader):
            b_x = x.to(device)
            # b_y = y.view((-1, h_dep * w_dep)).to(device)
            b_trans = trans.to(device)

            middle = cs_net(b_x)
            # output = u_net(middle)
            loss = loss_func(b_trans, middle.squeeze())
            # + loss_func(output.view((-1, h_dep*w_dep)), b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:', epoch, '|', 'Step:', step, '|', 'train loss:%.6f' % loss.data)

            if (step == 0): train_loss_list.append(loss.cpu().detach().numpy())

            if ((step + 1) % 100 == 0 or step == (NUM_OF_TRAINING_DATA // BATCH_SIZE) - 1):

                train_loss_list.append(loss.cpu().detach().numpy())

                print('--------------------------------')

                cs_net.eval()
                with torch.no_grad():

                    # t_loss = 0

                    # 随机从测试集抽取一个batch看训练效果
                    sample_seq = random_sample(0, NUM_OF_TEST_DATA // BATCH_SIZE, 1)  # 只有一个元素的list，用来随机选取某个batch展示其深度重建图

                    for seq, (test_data, test_trans, test_label) in enumerate(test_loader):

                        if (seq == sample_seq[0]):

                            t_x = test_data.to(device)
                            t_y = test_label.view((-1, h_dep * w_dep)).to(device)
                            t_trans = test_trans.to(device)

                            t_middle = cs_net(t_x)
                            loss_tmp = loss_func(t_trans, t_middle.squeeze()) \
                                # + loss_func(t_output.view((-1, h_dep*w_dep)), t_y)
                            # t_loss += loss_tmp.data * t_x.shape[0]

                            '''t_output仅做效果展示用'''
                            t_temp = t_middle
                            t_temp[t_temp < 0] = 0
                            for b in range(BATCH_SIZE):
                                t_temp[b] = normalization(t_temp[b])

                            t_output = u_net(t_temp)  # 注意CSNET学习的对象是transinet/RATE, RATE=K/rate，但在归一化时存在系数结果并无区别

                            # 抽取瞬态图的第256帧看恢复效果
                            selected_frame = time_bin // 2

                            # 随机展示测试数据效果
                            for j in range(t_x.shape[0]):
                                # original data (first row) for viewing
                                gt = t_y.data.cpu().numpy()[j]
                                a[0][j].clear()
                                ax0 = a[0][j].imshow(np.reshape(gt, (h_dep, w_dep)), cmap='gray')
                                a[0][j].set_xticks(())
                                a[0][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb0[j] = f.colorbar(ax0, ax=a[0][j])
                                else:
                                    cb0[j].update_normal(ax0)

                                # test output (second row) for viewing
                                rc = t_output.data.cpu().numpy()[j]
                                a[1][j].clear()
                                ax1 = a[1][j].imshow(np.reshape(rc, (h_dep, w_dep)), cmap='gray')
                                a[1][j].set_xticks(())
                                a[1][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb1[j] = f.colorbar(ax1, ax=a[1][j])
                                else:
                                    cb1[j].update_normal(ax1)

                                psnr = peak_signal_noise_ratio(gt, rc, data_range=1)
                                ssim = structural_similarity(gt, rc, data_range=1)
                                mse = mean_squared_error(gt, rc)
                                a[1][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (psnr, ssim, mse), fontsize=8)

                                # # selected frame from origin transient data
                                # fgt = t_trans.data.cpu().numpy()[j, selected_frame, :, :]
                                # a[2][j].clear()
                                # ax2 = a[2][j].imshow(np.reshape(fgt, (h_tr, w_tr)), cmap='gray')
                                # a[2][j].set_xticks(())
                                # a[2][j].set_yticks(())
                                # if epoch == 0 and step == (100-1):
                                #     cb2[j] = f.colorbar(ax2, ax=a[2][j])
                                # else:
                                #     cb2[j].update_normal(ax2)
                                #
                                # # selected frame from test output transient data
                                # frc = t_middle.data.cpu().numpy()[j, selected_frame, :, :]
                                # a[3][j].clear()
                                # ax3 = a[3][j].imshow(np.reshape(frc, (h_tr, w_tr)), cmap='gray')
                                # a[3][j].set_xticks(())
                                # a[3][j].set_yticks(())
                                # if epoch == 0 and step == (100-1):
                                #     cb3[j] = f.colorbar(ax3, ax=a[3][j])
                                # else:
                                #     cb3[j].update_normal(ax3)
                                #
                                # f_psnr = peak_signal_noise_ratio(fgt, frc, data_range=1)
                                # f_ssim = structural_similarity(fgt, frc, data_range=1)
                                # f_mse = mean_squared_error(fgt, frc)
                                # a[3][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (f_psnr, f_ssim, f_mse),
                                #                   fontsize=8)

                            plt.draw();plt.pause(1)

                            break

                        else:
                            continue

                    # print('Epoch:', epoch, '| test loss:%.6f' % (t_loss / NUM_OF_TEST_DATA))
                    # test_loss_list.append(t_loss.cpu() / NUM_OF_TEST_DATA)

                    print('Epoch:', epoch, '| test loss:%.6f' % loss_tmp)
                    test_loss_list.append(loss_tmp.cpu())
                    print('--------------------------------')

                del loss_tmp, t_middle \
                    , t_output, t_temp

                cs_net.train()

            del loss, middle \
                # , output

        torch.save(cs_net.state_dict(),
                   'trained_model/cs_net/3D/cr10/pure_cs_net_params_3D_epoch_%d_K_500.pth' % epoch)

        train_loss_full_list.append(train_loss_list)
        test_loss_full_list.append(test_loss_list)

    train_loss_full_list = np.array(train_loss_full_list)
    np.save('loss/cs_net/3D/cr10/pure_cs_3D_train_loss_K_500.npy', train_loss_full_list)

    test_loss_full_list = np.array(test_loss_full_list)
    np.save('loss/cs_net/3D/cr10/pure_cs_3D_test_loss_K_500.npy', test_loss_full_list)
    #
    print('net time: %4f' % (time.time() - net_start))


def CSNET_2D_train(train_dataset_len, test_dataset_len, BATCH_SIZE, EPOCH, SHOW_SAMPLE_NUM, h_tr, w_tr, device,
                   train_loader, test_loader, optimizer, loss_func, cs_net):

    # initialize figure
    f, a = plt.subplots(2, SHOW_SAMPLE_NUM, figsize=(50, 50))
    cb0, cb1 = {}, {}

    plt.ion()  # continuously plot

    net_start = time.time()
    train_loss_full_list, test_loss_full_list = [], []
    for epoch in range(EPOCH):

        train_loss_list, test_loss_list = [], []

        for step, (com, trans) in enumerate(train_loader):
            b_x = com.to(device)
            b_trans = trans.to(device)

            out = cs_net(b_x)
            # output = u_net(middle)
            loss = loss_func(b_trans.view(-1, (h_tr * w_tr)), out.view(-1, (h_tr * w_tr)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:', epoch, '|', 'Step:', step, '|', 'train loss:%f' % loss.data)

            if (step == 0): train_loss_list.append(loss.cpu().detach().numpy())

            if ((step + 1) % 100 == 0 or step == (train_dataset_len // BATCH_SIZE) - 1):

                train_loss_list.append(loss.cpu().detach().numpy())

                print('--------------------------------')

                cs_net.eval()
                with torch.no_grad():

                    sample_seq = random_sample(0, test_dataset_len // BATCH_SIZE, 1)  # 只有一个元素的list，用来随机选取某个batch展示其深度重建图
                    # print(sample_seq[0])
                    for seq, (test_com, test_trans) in enumerate(test_loader):

                        if (seq == sample_seq[0]):

                            t_x = test_com.to(device)
                            t_trans = test_trans.to(device)

                            t_out = cs_net(t_x)
                            t_loss = loss_func(t_trans.view(-1, (h_tr * w_tr)), t_out.view(-1, (h_tr * w_tr)))

                            # 随机展示测试数据效果
                            for j in range(SHOW_SAMPLE_NUM):
                                # original data (first row) for viewing
                                a[0][j].clear()
                                gt = np.reshape(t_trans.data.cpu().numpy()[j], (h_tr, w_tr))
                                ax0 = a[0][j].imshow(gt, cmap='gray')
                                a[0][j].set_xticks(())
                                a[0][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb0[j] = f.colorbar(ax0, ax=a[0][j])
                                else:
                                    cb0[j].update_normal(ax0)

                                # test output (second row) for viewing
                                a[1][j].clear()
                                rc = np.reshape(t_out.data.cpu().numpy()[j], (h_tr, w_tr))
                                ax1 = a[1][j].imshow(rc, cmap='gray')  # Variable 转 numpy需要加.data，可见2.2节
                                a[1][j].set_xticks(())
                                a[1][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb1[j] = f.colorbar(ax1, ax=a[1][j])
                                else:
                                    cb1[j].update_normal(ax1)

                                psnr = peak_signal_noise_ratio(gt, rc, data_range=1)
                                ssim = structural_similarity(gt, rc, data_range=1)
                                mse = mean_squared_error(gt, rc)
                                a[1][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (psnr, ssim, mse), fontsize=20)

                            plt.draw();plt.pause(0.05)

                            break

                        else:
                            continue

                    print('Epoch:', epoch, '| test loss:%f' % t_loss)
                    test_loss_list.append(t_loss.cpu())
                    print('--------------------------------')

                del t_loss, t_out

                cs_net.train()

            del loss, out

        torch.save(cs_net.state_dict(),
                   'trained_model/cs_net/2D/cr20/pure_cs_net_params_2D_epoch_%d_K_100.pth' % epoch)

        train_loss_full_list.append(train_loss_list)
        test_loss_full_list.append(test_loss_list)

    train_loss_full_list = np.array(train_loss_full_list)
    np.save('loss/cs_net/2D/cr20/pure_cs_2D_train_loss_K_100.npy', train_loss_full_list)

    test_loss_full_list = np.array(test_loss_full_list)
    np.save('loss/cs_net/2D/cr20/pure_cs_2D_test_loss_K_100.npy', test_loss_full_list)

    print('net time: %4f' % (time.time() - net_start))


def UNET_train(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, BATCH_SIZE, EPOCH, h_dep, w_dep, device,
                   train_loader, test_loader, optimizer, loss_func, u_net):

    # initialize figure
    f, a = plt.subplots(2, BATCH_SIZE, figsize=(50, 20))
    cb0, cb1 = {}, {}

    plt.ion()   # continuously plot

    net_start = time.time()
    train_loss_full_list, test_loss_full_list = [], []
    for epoch in range(EPOCH):

        train_loss_list, test_loss_list = [], []

        for step, (x, y) in enumerate(train_loader):
            b_x = x.to(device)
            b_y = y.view((-1, h_dep * w_dep)).to(device)

            output = u_net(b_x)
            loss = loss_func(output.view((-1, h_dep * w_dep)), b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:', epoch, '|', 'Step:', step, '|', 'train loss:%.6f' % loss.data)

            if (step == 0): train_loss_list.append(loss.cpu().detach().numpy())

            if ((step + 1) % 100 == 0 or step == (NUM_OF_TRAINING_DATA // BATCH_SIZE) - 1):

                train_loss_list.append(loss.cpu().detach().numpy())

                print('--------------------------------')

                u_net.eval()
                with torch.no_grad():

                    # t_loss = 0
                    sample_seq = random_sample(0, NUM_OF_TEST_DATA // BATCH_SIZE,
                                               1)  # 只有一个元素的list，用来随机选取某个batch展示其深度重建图
                    # print(sample_seq[0])
                    for seq, (test_data, test_label) in enumerate(test_loader):

                        if (seq == sample_seq[0]):

                            t_x = test_data.to(device)
                            t_y = test_label.to(device)

                            t_out = u_net(t_x)
                            loss_tmp = loss_func(t_out.view((-1, h_dep * w_dep)), t_y.view((-1, h_dep * w_dep)))
                            # t_loss += loss_tmp.data * t_x.shape[0]

                            # 随机展示测试数据效果
                            for j in range(t_x.shape[0]):
                                # original data (first row) for viewing
                                a[0][j].clear()
                                gt = np.reshape(t_y.data.cpu().numpy()[j], (h_dep, w_dep))
                                ax0 = a[0][j].imshow(gt, cmap='gray')
                                a[0][j].set_xticks(())
                                a[0][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb0[j] = f.colorbar(ax0, ax=a[0][j])
                                else:
                                    cb0[j].update_normal(ax0)

                                # test output (second row) for viewing
                                a[1][j].clear()
                                rc = np.reshape(t_out.data.cpu().numpy()[j], (h_dep, w_dep))
                                ax1 = a[1][j].imshow(rc, cmap='gray')  # Variable 转 numpy需要加.data，可见2.2节
                                a[1][j].set_xticks(())
                                a[1][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb1[j] = f.colorbar(ax1, ax=a[1][j])
                                else:
                                    cb1[j].update_normal(ax1)

                                psnr = peak_signal_noise_ratio(gt, rc, data_range=1)
                                ssim = structural_similarity(gt, rc, data_range=1)
                                mse = mean_squared_error(gt, rc)
                                a[1][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (psnr, ssim, mse), fontsize=8)

                            plt.draw();plt.pause(1)

                            break

                        else:
                            continue

                    del t_out

                    # print('Epoch:', epoch, '| test loss:%.6f' % (t_loss / NUM_OF_TEST_DATA))
                    # test_loss_list.append(t_loss.cpu() / NUM_OF_TEST_DATA)

                    print('Epoch:', epoch, '| test loss:%.6f' % loss_tmp)
                    test_loss_list.append(loss_tmp.cpu())
                    print('--------------------------------')

                u_net.train()

            del output, loss

        torch.save(u_net.state_dict(),
                   'trained_model/unet/unet_params_retrain_epoch_%d_K_100.pth' % epoch)

        train_loss_full_list.append(train_loss_list)
        test_loss_full_list.append(test_loss_list)

    train_loss_full_list = np.array(train_loss_full_list)
    np.save('loss/unet/unet_retrain_train_loss_K_100.npy', train_loss_full_list)

    test_loss_full_list = np.array(test_loss_full_list)
    np.save('loss/unet/unet_retrain_test_loss_K_100.npy', test_loss_full_list)

    print('net time: %4f' % (time.time() - net_start))


'''端到端联合训练'''
def CS_UNET_32_train(NUM_OF_TRAINING_DATA, NUM_OF_TEST_DATA, BATCH_SIZE, EPOCH, lamb,
                   h_dep, w_dep, device, train_loader, test_loader, optimizer, loss_func, net):

    # initialize figure
    f, a = plt.subplots(3, BATCH_SIZE, figsize=(50, 50))
    cb0, cb1, cb2 = {}, {}, {}

    plt.ion()  # continuously plot

    net_start = time.time()
    train_loss_full_list, test_loss_full_list = [], []
    for epoch in range(EPOCH):

        train_loss_list, test_loss_list = [], []

        for step, (x, trans, y) in enumerate(train_loader):
            b_x = x.to(device)
            b_trans = trans.to(device)
            b_y = y.view((-1, h_dep * w_dep)).to(device)

            middle, output = net(b_x)
            loss = loss_func(b_y, output.view((-1, h_dep * w_dep))) \
                 + loss_func(middle.squeeze(), b_trans) * lamb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:', epoch, '|', 'Step:', step, '|', 'train loss:%.6f' % loss.data)

            if (step == 0): train_loss_list.append(loss.cpu().detach().numpy())

            if ((step + 1) % 100 == 0 or step == (NUM_OF_TRAINING_DATA // BATCH_SIZE) - 1):

                train_loss_list.append(loss.cpu().detach().numpy())

                print('--------------------------------')

                net.eval()
                with torch.no_grad():

                    sample_seq = random_sample(0, NUM_OF_TEST_DATA // BATCH_SIZE, 1)

                    for seq, (test_data, test_trans, test_label) in enumerate(test_loader):

                        if (seq == sample_seq[0]):

                            t_x = test_data.to(device)
                            t_trans = test_trans
                            t_y = test_label.view((-1, h_dep * w_dep)).to(device)

                            t_middle, t_out = net(t_x)
                            t_middle = t_middle.squeeze()
                            t_loss = loss_func(t_out.view((-1, h_dep * w_dep)), t_y) \
                                   + loss_func(t_middle, t_trans) * lamb

                            for b in range(BATCH_SIZE):
                                t_middle[b] = normalization(t_middle[b])

                            u_net = UNET(32, 32, 512).to(device)
                            u_net.load_state_dict(
                                torch.load("trained_model/unet/unet_params_retrain_epoch_4.pth", map_location=device))

                            t_output = u_net(t_middle)

                            # 随机展示测试数据效果
                            for j in range(t_x.shape[0]):
                                # original data (first row) for viewing
                                a[0][j].clear()
                                gt = np.reshape(t_y.data.cpu().numpy()[j], (h_dep, w_dep))
                                ax0 = a[0][j].imshow(gt, cmap='gray')
                                a[0][j].set_xticks(())
                                a[0][j].set_yticks(())
                                if epoch == 0 and step == (100 - 1):
                                    cb0[j] = f.colorbar(ax0, ax=a[0][j])
                                else:
                                    cb0[j].update_normal(ax0)

                                # test output (second row) for viewing
                                a[1][j].clear()
                                rc = np.reshape(t_out.data.cpu().numpy()[j], (h_dep, w_dep))
                                ax1 = a[1][j].imshow(rc, cmap='gray')  # Variable 转 numpy需要加.data，可见2.2节
                                a[1][j].set_xticks(())
                                a[1][j].set_yticks(())
                                if epoch == 0 and step == (100-1):
                                    cb1[j] = f.colorbar(ax1, ax=a[1][j])
                                else:
                                    cb1[j].update_normal(ax1)

                                psnr = peak_signal_noise_ratio(gt, rc, data_range=1)
                                ssim = structural_similarity(gt, rc, data_range=1)
                                mse = mean_squared_error(gt, rc)
                                a[1][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (psnr, ssim, mse), fontsize=8)

                                ##############################################################
                                a[2][j].clear()
                                xrc = np.reshape(t_output.data.cpu().numpy()[j], (h_dep, w_dep))
                                ax2 = a[2][j].imshow(xrc, cmap='gray')  # Variable 转 numpy需要加.data，可见2.2节
                                a[2][j].set_xticks(())
                                a[2][j].set_yticks(())
                                if epoch == 0 and step == (100 - 1):
                                    cb2[j] = f.colorbar(ax2, ax=a[2][j])
                                else:
                                    cb2[j].update_normal(ax2)

                                xpsnr = peak_signal_noise_ratio(gt, xrc, data_range=1)
                                xssim = structural_similarity(gt, xrc, data_range=1)
                                xmse = mean_squared_error(gt, xrc)
                                a[2][j].set_title(label='PSNR=%f \n SSIM=%f \n MSE=%f' % (xpsnr, xssim, xmse), fontsize=8)

                            plt.draw();plt.pause(1)

                            break

                        else:
                            continue

                    print('Epoch:', epoch, '| test loss:%.6f' %t_loss)
                    test_loss_list.append(t_loss.cpu())
                    print('--------------------------------')

                    del t_out, t_loss

                net.train()

            del output, loss

        torch.save(net.state_dict(),
                   'trained_model/end2end/cr20/cs_unet_params_epoch_%d_lamb_%4f.pth' % (epoch+1, lamb))

        train_loss_full_list.append(train_loss_list)
        test_loss_full_list.append(test_loss_list)

    # train_loss_full_list = np.array(train_loss_full_list)
    # np.save('loss/end2end/cr20/cs_unet_train_loss_lamb_%f.npy'%lamb, train_loss_full_list)
    #
    # test_loss_full_list = np.array(test_loss_full_list)
    # np.save('loss/end2end/cr20/cs_unet_test_loss_lamb_%f.npy'%lamb, test_loss_full_list)

    print('net time: %4f' % (time.time() - net_start))