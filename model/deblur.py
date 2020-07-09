from model.sub_module.common import *
from model.sub_module.flownetS import *

"""
DVD-SFE:optical flow module, sharp feature extraction module, deblurring module
"""


def make_model(args):
    return Deblur(args)


class Deblur(nn.Module):
    def __init__(self, args):
        super(Deblur, self).__init__()

        self.args = args
        n_resblock = 3  # 3
        n_feats = 32  # 32
        kernel_size = 5
        self.n_colors = args.n_colors

        InBlock = [
            Conv(self.n_colors * (2 * args.num_frames_per_sequence - 1), n_feats, kernel_size, padding=2, act=True),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2)]

        blur_Encoder_first = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                              ResBlock(Conv, n_feats * 4, kernel_size, padding=2),
                              ResBlock(Conv, n_feats * 4, kernel_size, padding=2),
                              ResBlock(Conv, n_feats * 4, kernel_size, padding=2)]

        blur_Encoder_second = [Conv(n_feats * 6, n_feats * 8, kernel_size, padding=2, stride=2, act=True),
                               ResBlock(Conv, n_feats * 8, kernel_size, padding=2),
                               ResBlock(Conv, n_feats * 8, kernel_size, padding=2),
                               ResBlock(Conv, n_feats * 8, kernel_size, padding=2)]

        # decoder2
        Decoder_second = [ResBlock(Conv, n_feats * 8, kernel_size, padding=2) for _ in range(n_resblock)]
        Decoder_second.append(Deconv(n_feats * 8, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        Decoder_first = [ResBlock(Conv, n_feats * 8, kernel_size, padding=2) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(n_feats * 8, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))

        OutBlock = [ResBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in range(n_resblock)]
        OutBlock.append(Conv(n_feats * 4, self.n_colors, kernel_size, padding=2))

        # extract shallow features
        self.convD = Conv(n_feats, n_feats * 2, kernel_size, stride=2, padding=2,
                          act=True)
        self.convN = Conv(self.n_colors * args.num_frames_per_sequence, n_feats, kernel_size, stride=1, padding=2,
                          act=True)

        self.inBlock = nn.Sequential(*InBlock)
        self.blur_encoder_first = nn.Sequential(*blur_Encoder_first)
        self.blur_encoder_second = nn.Sequential(*blur_Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)

        print('use the flow_pretrained model from {}'.format(args.flowModel_path))
        self.get_flow = flownets(data=torch.load(args.flowModel_path))

    def forward(self, blur, sharp):
        blur_list = list(torch.split(blur, 3, dim=1))
        blur_middle = blur_list[len(blur_list) // 2]

        cat_pre_mid = torch.cat((blur_middle, blur_list[0]), dim=1)
        cat_next_mid = torch.cat((blur_middle, blur_list[-1]), dim=1)
        flow_1 = self.get_flow(cat_pre_mid) * 20
        flow_2 = self.get_flow(cat_next_mid) * 20
        flow_1 = F.interpolate(flow_1, size=blur_middle.shape[-2:], mode='bilinear', align_corners=False)
        flow_2 = F.interpolate(flow_2, size=blur_middle.shape[-2:], mode='bilinear', align_corners=False)

        warp_pre = warp(blur_list[0], flow_1)
        warp_next = warp(blur_list[-1], flow_2)

        new_input = torch.cat((blur_list[0] , warp_pre, blur_middle, warp_next , blur_list[-1]), dim=1)

        blur_inblock = self.inBlock(new_input)  # 15--> 32
        sharp_un = self.convN(sharp)  # 9-->32
        sharp_down = self.convD(sharp_un)  # 32-->64

        blur_encoder_first = self.blur_encoder_first(torch.cat((blur_inblock, sharp_un), dim=1))  # 64--> 128

        blur_encoder_second = self.blur_encoder_second(
            torch.cat((blur_encoder_first, sharp_down), dim=1))  # 128+64--> 256

        blur_decoder_second = self.decoder_second(blur_encoder_second)  # 256-->64

        blur_decoder_first = self.decoder_first(
            torch.cat((blur_decoder_second, blur_encoder_first, sharp_down), dim=1))  # 64+128+64--> 64

        blur_outBlock = self.outBlock(torch.cat((blur_decoder_first, blur_inblock, sharp_un), dim=1))  # 64+32+32-->3

        return blur_outBlock, warp_pre, warp_next, flow_2, flow_1
