import os
from data import vddata


class Benchmark_video(vddata.VD_Data):
    """
    Data generator for benchmark tasks
    """

    def __init__(self, args, name='', train=False):
        super(Benchmark_video, self).__init__(
            args, name=name, train=train
        )

    def _set_filesystem(self, dir_data):

        self.apath = os.path.join(dir_data, self.name)
        if not self.args.test_only:
            self.dir_video = os.path.join(self.apath, "val")
            print("validation video path :", self.dir_video)
        else:
            self.dir_video = os.path.join(self.apath, 'test')
            print("test video path :", self.dir_video)
