import os
import glob

from data import common

import numpy as np
import imageio
import random
import torch
import torch.utils.data as data


class VD_Data(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.num_sequence_per_video = args.num_sequence_per_video
        self.num_frames_per_sequence = args.num_frames_per_sequence

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_sharp, self.images_blur, self.images_gt = self._scan()

        self.num_video = len(self.images_sharp)
        print("Number of videos to load:", self.num_video)

        if train:
            self.repeat = args.test_every // max((self.num_video // self.args.batch_size), 1)

        if args.process:
            self.data_sharp, self.data_blur, self.data_gt = self._load(self.num_video)

    def _set_filesystem(self, dir_data):
        pass

    def _scan(self):
        """
        Returns a list of image directories
        """
        video_names = sorted(glob.glob(os.path.join(self.dir_video, "*")))
        print("total video:", len(video_names))

        names_sharp, names_blur, names_gt = [], [], []
        if self.train:
            for idx in range(len(video_names)):
                if idx == len(video_names) - 1:
                    video_sharp_frame = sorted(glob.glob(os.path.join(video_names[0], "GT", "*")))
                else:
                    video_sharp_frame = sorted(glob.glob(os.path.join(video_names[idx + 1], "GT", "*")))
                video_blur_frame = sorted(glob.glob(os.path.join(video_names[idx], "input", "*")))
                video_GT_frame = sorted(glob.glob(os.path.join(video_names[idx], "GT", "*")))
                assert len(video_blur_frame) == len(video_GT_frame)

                if len(video_sharp_frame) < len(video_GT_frame):
                    temp = video_sharp_frame[-1]
                    pre_sharp_length = len(video_sharp_frame)
                    for _num in range(len(video_GT_frame) - pre_sharp_length):
                        video_sharp_frame.append(temp)
                assert len(video_sharp_frame) >= len(video_GT_frame)
                # split_idx = min(len(video_sharp_frame), len(video_blur_frame), len(video_GT_frame))
                names_sharp.append(video_sharp_frame)
                names_blur.append(video_blur_frame)
                names_gt.append(video_GT_frame)
                self.n_frames_video.append(len(video_blur_frame))

            return names_sharp, names_blur, names_gt
        else:
            for idx in range(len(video_names)):
                if idx == len(video_names) - 1:
                    video_sharp_frame = sorted(glob.glob(os.path.join(video_names[0], "GT", "*")))
                else:
                    video_sharp_frame = sorted(glob.glob(os.path.join(video_names[idx + 1], "GT", "*")))
                video_blur_frame = sorted(
                    glob.glob(os.path.join(video_names[idx], "input", "*")))
                video_GT_frame = sorted(glob.glob(os.path.join(video_names[idx], "GT", "*")))
                assert len(video_blur_frame) == len(video_GT_frame)

                if len(video_sharp_frame) < len(video_GT_frame):
                    temp = video_sharp_frame[-1]
                    pre_sharp_length = len(video_sharp_frame)
                    for _num in range(len(video_GT_frame) - pre_sharp_length):
                        video_sharp_frame.append(temp)
                assert len(video_sharp_frame) >= len(video_GT_frame)

                names_sharp.append(video_sharp_frame)
                names_blur.append(video_blur_frame)
                names_gt.append(video_GT_frame)
                self.n_frames_video.append(len(video_blur_frame))

            return names_sharp, names_blur, names_gt

    # 将每个视频中所有图像都读取到，返回的 data_sharp.shape(num_videos, n_frames_per_video, H, W, C)
    def _load(self, n_videos):
        data_blur, data_sharp, data_gt = [], [], []
        for idx in range(n_videos):
            sharps = np.array([imageio.imread(sharp_name) for sharp_name in self.images_sharp[idx]])
            blurs = np.array([imageio.imread(blur_name) for blur_name in self.images_blur[idx]])
            gts = np.array([imageio.imread(gt_name) for gt_name in self.images_gt[idx]])

            data_blur.append(blurs)
            data_sharp.append(sharps)
            data_gt.append(gts)

        return data_sharp, data_blur, data_gt

    def __getitem__(self, idx):
        if self.args.process:
            blurs, sharps, gts, filenames = self._load_file_from_loaded_data(idx)
        else:
            blurs, sharps, gts, filenames = self._load_file(idx)

        blurs_list = [blurs[i] for i in range(blurs.shape[0])]
        blurs = np.concatenate(blurs_list, axis=-1)

        sharp_list = [sharps[i] for i in range(sharps.shape[0])]
        sharps = np.concatenate(sharp_list, axis=-1)

        gt_list = [gts[i] for i in range(gts.shape[0])]
        gts = np.concatenate(gt_list, axis=-1)

        patches = [self.get_patch(blurs, sharps, gts)]
        blurs = np.array([patch[0] for patch in patches])
        sharps = np.array([patch[1] for patch in patches])
        gts = np.array([patch[2] for patch in patches])

        blurs = torch.cat(torch.split(torch.from_numpy(blurs), 3, dim=-1), dim=0).numpy()
        sharps = torch.cat(torch.split(torch.from_numpy(sharps), 3, dim=-1), dim=0).numpy()
        gts = torch.cat(torch.split(torch.from_numpy(gts), 3, dim=-1), dim=0).numpy()

        blur_tensors = common.np2Tensor(*blurs, rgb_range=self.args.rgb_range)
        sharp_tensors = common.np2Tensor(*sharps, rgb_range=self.args.rgb_range)
        gt_tensors = common.np2Tensor(*gts, rgb_range=self.args.rgb_range)

        return torch.stack(blur_tensors), torch.stack(sharp_tensors), torch.stack(gt_tensors), filenames

    def __len__(self):
        if self.train:
            return len(self.images_sharp) * self.repeat
        else:
            # if test, call all possible video sequence fragments
            return sum(self.n_frames_video) - (self.num_frames_per_sequence - 1) * len(self.n_frames_video)

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_video
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array and list of corresponding filenames
        """
        idx = self._get_index(idx)
        if self.train:
            f_sharps = self.images_sharp[idx]
            f_blurs = self.images_blur[idx]
            f_gts = self.images_gt[idx]
            start = random.randint(0, self.n_frames_video[idx] - self.num_frames_per_sequence)
            filenames = [os.path.splitext(os.path.basename(file))[0] for file in
                         f_gts[start:start + self.num_frames_per_sequence]]

            sharps = np.array(
                [imageio.imread(sharp_name) for sharp_name in f_sharps[start:start + self.num_frames_per_sequence]])
            blurs = np.array(
                [imageio.imread(blur_name) for blur_name in f_blurs[start:start + self.num_frames_per_sequence]])
            gts = np.array([imageio.imread(gt_name) for gt_name in f_gts[start:start + self.num_frames_per_sequence]])

        else:
            n_poss_frames = [n - self.num_frames_per_sequence + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_sharps = self.images_sharp[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            f_blurs = self.images_blur[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            filenames = [os.path.basename(file.split("/GT")[0]) + '.' + os.path.splitext(os.path.basename(file))[0] for
                         file in f_gts]
            sharps = np.array([imageio.imread(sharp_name) for sharp_name in f_sharps])
            blurs = np.array([imageio.imread(blur_name) for blur_name in f_blurs])
            gts = np.array([imageio.imread(gt_name) for gt_name in f_gts])

        return blurs, sharps, gts, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        if self.train:
            start = self._get_index(random.randint(0, self.n_frames_video[idx] - self.num_frames_per_sequence))
            sharps = self.data_sharp[idx][start:start + self.num_frames_per_sequence]
            blurs = self.data_blur[idx][start:start + self.num_frames_per_sequence]
            gts = self.data_gt[idx][start:start + self.num_frames_per_sequence]
            filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.images_gt[idx]]

        else:
            n_poss_frames = [n - self.num_frames_per_sequence + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            sharps = self.data_sharp[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            blurs = self.data_blur[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            gts = self.data_gt[video_idx][frame_idx:frame_idx + self.num_frames_per_sequence]
            filenames = [os.path.basename(file.split("/GT")[0]) + '.' + os.path.splitext(os.path.basename(file))[0] for
                         file in f_gts]

        return blurs, sharps, gts, filenames

    def get_patch(self, blur, sharp, gt):

        if self.train:
            blur, sharp, gt = common.get_patch(
                blur,
                sharp,
                gt,
                patch_size=self.args.patch_size,
            )
            if not self.args.no_augment:
                blur, sharp, gt = common.augment(blur, sharp, gt)
        else:
            ih, iw = blur.shape[:2]
            blur = blur[:ih, :iw]
            sharp = sharp[:ih, :iw]
            gt = gt[:ih, :iw]
        return blur, sharp, gt
