#coding=UTF-8


from torch import nn
from func import *
import torch.nn.functional as F
from UNET_3D_2D_parts import *


class CSNET_2D(nn.Module):
    def __init__(self, h_tr, w_tr, num_of_mask, bilinear=True):
        super(CSNET_2D, self).__init__()

        self.num_of_mask = num_of_mask
        self.h_tr = h_tr
        self.w_tr = w_tr

        self.fc1 = nn.Linear(self.num_of_mask, self.h_tr * self.w_tr)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.conv1 = nn.Conv2d(1, 64, 11, 1, padding=5)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        self.conv2 = nn.Conv2d(64, 32, 1, 1, padding=0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
        self.conv3 = nn.Conv2d(32, 1, 7, 1, padding=3)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.1)
        self.conv4 = nn.Conv2d(1, 64, 11, 1, padding=5)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
        self.conv5 = nn.Conv2d(64, 32, 1, 1, padding=0)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.1)
        self.conv6 = nn.Conv2d(32, 1, 7, 1, padding=3)
        nn.init.normal_(self.conv6.weight, mean=0, std=0.1)

        # # dnCNN-denoiser
        # features = 64
        # num_of_layers = 17
        # layers = []
        # layers.append(nn.Conv2d(1, features, 3, 1, padding=1))
        # layers.append(nn.ReLU(inplace=True))
        # for _ in range(num_of_layers - 2):
        #     layers.append(
        #         nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        #     layers.append(nn.BatchNorm2d(features))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=3, padding=1, bias=False))
        #
        # self.denoiser = nn.Sequential(*layers)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, self.h_tr, self.w_tr)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        # x = self.denoiser(x)

        return x


class CSNET_3D(nn.Module):
    def __init__(self, num_of_mask, h_tr, w_tr, bilinear=True):
        super(CSNET_3D, self).__init__()

        self.num_of_mask = num_of_mask
        self.h_tr = h_tr
        self.w_tr = w_tr

        self.fc1 = nn.Linear(self.num_of_mask, self.h_tr * self.w_tr)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.conv1 = nn.Conv3d(1, 64, (1, 11, 11), stride=1, padding=(0, 5, 5))
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        self.conv2 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
        self.conv3 = nn.Conv3d(32, 1, (1, 7, 7), stride=1, padding=(0, 3, 3))
        nn.init.normal_(self.conv3.weight, mean=0, std=0.1)
        self.conv4 = nn.Conv3d(1, 64, (1, 11, 11), stride=1, padding=(0, 5, 5))
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
        self.conv5 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=0)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.1)
        self.conv6 = nn.Conv3d(32, 1, (1, 7, 7), stride=1, padding=(0, 3, 3))
        nn.init.normal_(self.conv6.weight, mean=0, std=0.1)


    def forward(self, x):
        x = x.transpose(1, 2)  # (N, time_bin, num_of_mask)
        x = self.fc1(x)
        x = rearrange(x, 'b (c f) (h w) -> b c f h w', c=1, h=self.h_tr)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = x.squeeze(dim=1)
        # print(x.shape)

        return x


class UNET(nn.Module):
    def __init__(self, h_tr, w_tr, time_bin, bilinear=True):
        super(UNET, self).__init__()

        self.time_bin = time_bin
        self.h_tr = h_tr
        self.w_tr = w_tr

        self.preprocess = nn.AvgPool3d((4, 1, 1))
        self.inc = DoubleConv3D(1, 48)
        self.down1 = Down(48, 96)
        self.down2 = Down(96, 192)
        self.down3 = Down(192, 384)
        self.down4 = nn.AvgPool3d((2, 2, 2))
        self.conv2d = DoubleConv2D(384, 768, kernel_size=1, padding=0)

        self.up1 = Up(768, 384, frame=2, bilinear=bilinear)
        self.up2 = Up(384, 192, frame=8, bilinear=bilinear)
        self.up3 = Up(192, 96, frame=32, bilinear=bilinear)
        self.up4 = Up(96, 48, frame=128, bilinear=bilinear)

        self.regressor = nn.Sequential(OrderedDict([
            # ('conv1', nn.Conv2d(48, 1, kernel_size=1),),
            # ('relu1', nn.ReLU()),
            # ('flatten', nn.Flatten()),
            # ('linear1', nn.Linear(h*w, h*w)),
            # ('relu2', nn.ReLU()),
            # ('linear2', nn.Linear(h*w, h*w)),
            # ('relu3', nn.ReLU())

            ('upsam1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),  # 根据空间分辨率自行调整加不加这行
            ('norm', nn.BatchNorm2d(48)),
            ('upsam2', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('conv1', nn.Conv2d(48, 1, kernel_size=1)),
            ('norm1', nn.BatchNorm2d(1)),
            ('relu1', nn.ReLU(inplace=True)),  # 当设置为True时，我们在通过relu()计算时的得到的新值不会占用新的空间而是直接覆盖原来的值
            # ('conv2', nn.Conv2d(1, h_dep*w_dep, kernel_size=(h_dep, w_dep))),
            # ('norm2', nn.BatchNorm2d(h_dep*w_dep)),
            # ('relu2', nn.ReLU(inplace=True)),
            # ('batch2', nn.BatchNorm2d(h_dep*w_dep)),
            # ('conv3', nn.Conv2d(64*64, 64*64, kernel_size=1)),
            # ('relu3', nn.ReLU()),

            ('flat', nn.Flatten())
        ]))

    def forward(self, x):

        x = torch.reshape(x, (-1, 1, self.time_bin, self.h_tr, self.w_tr))  # (N, 1, 512, 64, 64)
        x = self.preprocess(x)  # N, 1, 128, 64, 64

        x1 = self.inc(x)  # 由于需要3D卷积所以保留帧数维，x参与UNET，将middle作为等效的瞬态矩阵参与计算loss   (N, 48, 256, h, w)  N, 48, 512, 64, 64
        x2 = self.down1(x1)  # (N, 96, 64, h/2, w/2) N, 96, 128, 32, 32
        x3 = self.down2(x2)  # (N, 192, 16, h/4, w/4) N,192, 32, 16, 16
        x4 = self.down3(x3)  # (N, 384, 4, h/8, w/8) N, 384, 8, 8, 8
        x5 = self.down4(x4)  # (N, 384, 1, h/16, w/16) N, 384, 1, 4, 4
        x5 = x5.squeeze(dim=2)  # (N, 384, h/16, w/16) N, 384, 4, 4
        x5 = self.conv2d(x5)  # (N, 768, h/16, w/16) N, 768, 4, 4

        x6 = self.up1(x5,
                      x4)  # (N, 768, h/16, w/16)->(N, 384, h/8, w/8)+(N, 384, h/8, w/8)->(N, 768, h/8, w/8)->(N, 384, h/8, h/8)  N, 384, 8, 8
        del x4, x5
        x7 = self.up2(x6,
                      x3)  # (N, 384, h/8, w/8)->(N, 192, h/4, w/4)+(N, 192, h/4, w/4)->(N, 384, h/4, w/4)->(N, 192, h/4, h/4)  N, 192, 16, 16
        del x3, x6
        x8 = self.up3(x7,
                      x2)  # (N, 192, h/4, w/4)->(N, 96, h/2, w/2)+(N, 96, h/2, w/2)->(N, 192, h/2, w/2)->(N, 96, h/2, h/2)  N, 96, 32, 32
        del x2, x7
        x9 = self.up4(x8,
                      x1)  # (N, 96, h/2, w/2)->(N, 48, h, w)+(N, 48, h, w)->(N, 96, h, w)->(N, 48, h, w)  N, 48, 64, 64
        del x1, x8

        logits = self.regressor(x9)  # (N, h*w)
        del x9

        return logits


'''联合训练端到端'''
class CS_UNET(nn.Module):
    def __init__(self, num_of_mask, h_tr, w_tr, time_bin, bilinear=True):
        super(CS_UNET, self).__init__()

        self.num_of_mask = num_of_mask
        self.time_bin = time_bin
        self.h_tr = h_tr
        self.w_tr = w_tr

        self.fc1 = nn.Linear(self.num_of_mask, self.h_tr * self.w_tr)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)

        self.conv1 = nn.Conv3d(1, 64, (1, 11, 11), stride=1, padding=(0, 5, 5))
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        self.conv2 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
        self.conv3 = nn.Conv3d(32, 1, (1, 7, 7), stride=1, padding=(0, 3, 3))
        nn.init.normal_(self.conv3.weight, mean=0, std=0.1)
        self.conv4 = nn.Conv3d(1, 64, (1, 11, 11), stride=1, padding=(0, 5, 5))
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
        self.conv5 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=0)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.1)
        self.conv6 = nn.Conv3d(32, 1, (1, 7, 7), stride=1, padding=(0, 3, 3))
        nn.init.normal_(self.conv6.weight, mean=0, std=0.1)


        self.layer_norm = nn.LayerNorm([1, time_bin, h_tr, w_tr], elementwise_affine=True)
        self.preprocess = nn.AvgPool3d((4, 1, 1))

        self.inc = DoubleConv3D(1, 48)
        self.down1 = Down(48, 96)
        self.down2 = Down(96, 192)
        self.down3 = Down(192, 384)
        self.down4 = nn.AvgPool3d((2, 2, 2))
        self.conv2d = DoubleConv2D(384, 768, kernel_size=1, padding=0)

        self.up1 = Up(768, 384, frame=2, bilinear=bilinear)
        self.up2 = Up(384, 192, frame=8, bilinear=bilinear)
        self.up3 = Up(192, 96, frame=32, bilinear=bilinear)
        self.up4 = Up(96, 48, frame=128, bilinear=bilinear)

        self.regressor = nn.Sequential(OrderedDict([
            # ('conv1', nn.Conv2d(48, 1, kernel_size=1),),
            # ('relu1', nn.ReLU()),
            # ('flatten', nn.Flatten()),
            # ('linear1', nn.Linear(h*w, h*w)),
            # ('relu2', nn.ReLU()),
            # ('linear2', nn.Linear(h*w, h*w)),
            # ('relu3', nn.ReLU())

            ('upsam1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),  # 根据空间分辨率自行调整加不加这行
            ('norm', nn.BatchNorm2d(48)),
            ('upsam2', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('conv1', nn.Conv2d(48, 1, kernel_size=1)),
            ('norm1', nn.BatchNorm2d(1)),
            ('relu1', nn.ReLU(inplace=True)),  # 当设置为True时，我们在通过relu()计算时的得到的新值不会占用新的空间而是直接覆盖原来的值
            # ('conv2', nn.Conv2d(1, h_dep*w_dep, kernel_size=(h_dep, w_dep))),
            # ('norm2', nn.BatchNorm2d(h_dep*w_dep)),
            # ('relu2', nn.ReLU(inplace=True)),
            # ('batch2', nn.BatchNorm2d(h_dep*w_dep)),
            # ('conv3', nn.Conv2d(64*64, 64*64, kernel_size=1)),
            # ('relu3', nn.ReLU()),

            ('flat', nn.Flatten())
        ]))

    def forward(self, x):

        x = x.transpose(1, 2)  # (N, time_bin, num_of_mask)
        x = self.fc1(x)
        x = rearrange(x, 'b (c f) (h w) -> b c f h w', c=1, h=self.h_tr)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        middle = F.relu(self.conv6(x))

        x = self.layer_norm(middle)
        x = self.preprocess(x)  # N, 1, 128, 64, 64

        x1 = self.inc(x)  # 由于需要3D卷积所以保留帧数维，x参与UNET，将middle作为等效的瞬态矩阵参与计算loss   (N, 48, 256, h, w)  N, 48, 512, 64, 64
        x2 = self.down1(x1)  # (N, 96, 64, h/2, w/2) N, 96, 128, 32, 32
        x3 = self.down2(x2)  # (N, 192, 16, h/4, w/4) N,192, 32, 16, 16
        x4 = self.down3(x3)  # (N, 384, 4, h/8, w/8) N, 384, 8, 8, 8
        x5 = self.down4(x4)  # (N, 384, 1, h/16, w/16) N, 384, 1, 4, 4
        x5 = x5.squeeze(dim=2)  # (N, 384, h/16, w/16) N, 384, 4, 4
        x5 = self.conv2d(x5)  # (N, 768, h/16, w/16) N, 768, 4, 4

        x6 = self.up1(x5, x4)  # (N, 768, h/16, w/16)->(N, 384, h/8, w/8)+(N, 384, h/8, w/8)->(N, 768, h/8, w/8)->(N, 384, h/8, h/8)  N, 384, 8, 8
        del x4, x5
        x7 = self.up2(x6, x3)  # (N, 384, h/8, w/8)->(N, 192, h/4, w/4)+(N, 192, h/4, w/4)->(N, 384, h/4, w/4)->(N, 192, h/4, h/4)  N, 192, 16, 16
        del x3, x6
        x8 = self.up3(x7, x2)  # (N, 192, h/4, w/4)->(N, 96, h/2, w/2)+(N, 96, h/2, w/2)->(N, 192, h/2, w/2)->(N, 96, h/2, h/2)  N, 96, 32, 32
        del x2, x7
        x9 = self.up4(x8, x1)  # (N, 96, h/2, w/2)->(N, 48, h, w)+(N, 48, h, w)->(N, 96, h, w)->(N, 48, h, w)  N, 48, 64, 64
        del x1, x8

        logits = self.regressor(x9)  # (N, h*w)
        del x9

        return middle, logits