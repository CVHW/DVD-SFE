import decimal
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm
from utils.utils import *


class Trainer_VD:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args

        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp

        self.error_last = 1e8

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}

        if self.args.model.endswith('flow'):
            # 剔除flow parameter
            ignore_parameter = list(map(id, self.model.model.get_flow.parameters()))
            deblur_parameter = filter(lambda p: id(p) not in ignore_parameter, self.model.model.parameters())

            flow_parameter = self.model.model.get_flow.parameters()

            return optim.Adam(
                [{'params': deblur_parameter},
                 {'params': flow_parameter, 'lr': 1e-6}], **kwargs
            )

        else:
            return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        print("Video Deblur Training...")
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()

        for batch, (blur, sharp, gt, _) in enumerate(self.loader_train):
            blur_list = list(torch.split(blur, 1, dim=1))
            blur_dim_change = [torch.squeeze(blur, 1) for blur in blur_list]
            blur = torch.cat(blur_dim_change, 1)

            sharp_list = list(torch.split(sharp, 1, dim=1))
            sharp_dim_change = [torch.squeeze(sharp, 1) for sharp in sharp_list]
            sharp = torch.cat(sharp_dim_change, 1)

            gt = gt[:, int(gt.shape[1] / 2), :, :, :]  # GT 取中间帧

            blur = blur.to(self.device)
            sharp = sharp.to(self.device)
            gt = gt.to(self.device)

            self.optimizer.zero_grad()

            deblur, _, _, _, _ = self.model(blur, sharp)

            loss = self.loss(deblur, gt)

            self.ckp.report_log(loss.item())
            loss.backward()

            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.loss.display_loss(batch)))

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (blur, sharp, gt, filename) in enumerate(tqdm_test):
                blur_list = list(torch.split(blur, 1, dim=1))
                blur_dim_change = [torch.squeeze(blur, 1) for blur in blur_list]
                blur = torch.cat(blur_dim_change, 1)

                gt = gt[:, int(gt.shape[1] / 2), :, :, :]  # GT 取中间帧

                sharp_list = list(torch.split(sharp, 1, dim=1))
                sharp_dim_change = [torch.squeeze(sharp, 1) for sharp in sharp_list]
                sharp = torch.cat(sharp_dim_change, 1)

                blur = blur.to(self.device)
                sharp = sharp.to(self.device)
                gt = gt.to(self.device)

                deblur, warp_pre, warp_next, flow2_1, flow0_1 = self.model(blur, sharp)
                gt, deblur, warp_pre, warp_next = postprocess(gt, deblur, warp_pre, warp_next,
                                                              rgb_range=self.args.rgb_range)

                PSNR = cal_psnr(self.args, deblur, gt)

                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    save_list = [deblur, warp_pre, warp_next]
                    flow_list = [flow0_1, flow2_1]
                    self.ckp.save_flow(filename, flow_list)
                    self.ckp.save_images(filename, save_list)

        self.ckp.end_log(len(self.loader_test), train=False)
        best = self.ckp.psnr_log.max(0)

        self.ckp.write_log('[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
            self.args.data_test, self.ckp.psnr_log[-1],
            best[0], best[1] + 1))
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
