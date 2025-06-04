import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from lib.res2net_v1b_base import Res2Net_model
from lib.Swin_V2 import SwinTransformerV2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x




class GLFD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GLFD, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
        )

        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 1)
        self.conv_cat1 = BasicConv2d(2 * out_channel, out_channel, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.bran1 = BasicConv2d(in_channel, in_channel, 3, padding=1)

    def forward(self, x):

        m = self.bran1(x)
        
        x0 = self.branch0(m)
        x1 = self.branch1(m)
        x2 = self.branch2(m)

        xl = self.local_att(m)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        xg = (x_cat + self.conv_res(m))
        xl = (xl + self.conv_res(m))
        
        xxl0 = xg + xl
        xxl1 = xg * xl
        xg0 = self.conv_cat1(torch.cat((xxl0, xxl1), 1))
        x = xg0 + self.conv_res(m)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

#Information fusion module
class IF(nn.Module):
    def __init__(self, dim):
        super(IF, self).__init__()

        self.pool_h = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_attention = ChannelAttention(2*dim)
        self.down2 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.PReLU()
        )

    def forward(self, r, d):

        rd = torch.cat((d, r), dim=1)
        rd_ca = self.channel_attention(rd)
        rd_ca = self.down2(rd_ca)
        rd_ca0 = rd_ca * d

        rd_ca = rd_ca0 + r
        x_h = self.pool_h(rd_ca)
        x_w = self.pool_w(rd_ca)


        rd_caM = rd_ca0 * x_h
        rd_caA = rd_ca0 * x_w

        rdr00 = rd_caM + rd_ca
        D0    = r + rdr00

        rdr01 = rd_caA + rd_ca
        D1    = r + rdr01

        D = (D0 + D1)
        return D


###############################################################################

class ADINet(nn.Module):
    def __init__(self, channel=32,ind=50):
        super(ADINet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        #Backbone model
        self.swin_rgb = SwinTransformerV2()
        self.layer_dep  = Res2Net_model(ind)
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)

        ####################################################
        ## new change channle
        ####################################################
        self.fu_1c = nn.Conv2d(256, 128, 1)
        self.fu_2c = nn.Conv2d(512, 256, 1)
        self.fu_3c = nn.Conv2d(1024, 512, 1)
        self.fu_4c = nn.Conv2d(2048, 1024, 1)

        ###############################################
        # information fusion
        ###############################################
        self.if1 = IF(128)
        self.if2 = IF(256)
        self.if3 = IF(512)
        self.if4 = IF(1024)

        ###############################################
        # decoders fusion
        ###############################################
        self.glfd_4    = GLFD(1024,  channel)
        self.glfd_3    = GLFD(512+channel,  channel)
        self.glfd_2    = GLFD(256+channel,  channel)
        self.glfd_1    = GLFD(128+channel,  channel)

        self.ful_out = nn.Conv2d(channel, 1, 1)

        if self.training:
            self.initialize_weights()
                
    def forward(self, imgs, depths):

        stage_rgb = self.swin_rgb(imgs)
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(self.layer_dep0(depths))

        img_0 = stage_rgb[0]           #[b,128,64,64]
        img_1 = stage_rgb[1]           #[b,256,32,32]
        img_2 = stage_rgb[2]           #[b,512,16,16]
        # img_3 = stage_rgb[3]         #[b,1024,8,8]
        img_4 = stage_rgb[4]           #[b,1024,8,8]

        # ####################################################
        # new change channle
        # ####################################################
        dep_1 = self.fu_1c(dep_1)
        dep_2 = self.fu_2c(dep_2)
        dep_3 = self.fu_3c(dep_3)
        dep_4 = self.fu_4c(dep_4)

        ###############################################
        # information fusion
        ###############################################
        f_1 = self.if1(img_0, dep_1)
        f_2 = self.if2(img_1, dep_2)
        f_3 = self.if3(img_2, dep_3)
        f_4 = self.if4(img_4, dep_4)

        ful_0 =f_1
        ful_1 =f_2
        ful_2 =f_3
        ful_4 =f_4

        ####################################################
        ## decoder fusion
        ####################################################
        xf_4 = self.glfd_4(ful_4)
        xf_3_cat = torch.cat([ful_2, self.upsample_2(xf_4)], dim=1)

        xf_32 = self.glfd_3(xf_3_cat)
        xf_2_cat = torch.cat([ful_1, self.upsample_2(xf_32)], dim=1)

        xf_22 = self.glfd_2(xf_2_cat)
        xf_1_cat = torch.cat([ful_0, self.upsample_2(xf_22)], dim=1)

        xf_12 = self.glfd_1(xf_1_cat)
        ful_out = self.upsample_4(self.ful_out(xf_12))

        return ful_out

    def initialize_weights(self):  # 加载预训练模型权重，做初始化
        self.swin_rgb.load_state_dict(
             torch.load('./pre/swinv2_base_patch4_window16_256.pth')['model'],                #导入预训练模型地址
             strict=False)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


if __name__ == '__main__':
    from thop import profile

    rgb = torch.rand((2, 3, 256, 256)).cuda()
    depth = torch.rand((2, 1, 256, 256)).cuda()
    model = ADINet(32, 50).cuda()
    flops1, params1 = profile(model, inputs=(rgb, depth))

    print('params:%.2f(M)' % ((params1) / 1000000))
    print('flops:%.2f(G)' % ((flops1) / 1000000000))
    l = model(rgb, depth)
    print(l.size())