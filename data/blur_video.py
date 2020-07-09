import os
from data import vddata


# Data loader for blur videos
class Blur_Video(vddata.VD_Data):

    def __init__(self, args, name='BLUR_VIDEO', train=True):
        super(Blur_Video, self).__init__(args, name=name, train=train)

    def _scan(self):
        names_sharp, names_blur, names_gt = super(Blur_Video, self)._scan()

        return names_sharp, names_blur, names_gt

    def _set_filesystem(self, dir_data):
        print("loading video...")
        self.apath = os.path.join(dir_data, self.name)
        self.dir_video = os.path.join(self.apath, "train")
        print("Training video path:", self.dir_video)
