from importlib import import_module

from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        list_benchmarks_video = [args.data_test]
        benchmark_video = self.data_test in list_benchmarks_video

        # load training train_dataset
        if not self.args.test_only:

            m_train = import_module('data.blur_video')
            trainset = getattr(m_train, "Blur_Video")(self.args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu,
                num_workers=self.args.n_threads
            )
        else:
            self.loader_train = None

        if benchmark_video:
            m_test = import_module('data.benchmark_video')
            testset = getattr(m_test, 'Benchmark_video')(self.args, name=args.data_test, train=False)

        # load testing train_dataset
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.args.cpu,
            num_workers=self.args.n_threads
        )
