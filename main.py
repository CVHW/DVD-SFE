import torch

import data
import model
import loss
from options.option import args
from trainer.trainer_vd import Trainer_VD
from logger import logger

torch.manual_seed(args.seed)

loader = data.Data(args)

chkp = logger.Logger(args)
model = model.Model(args, chkp)
loss = loss.Loss(args, chkp) if not args.test_only else None
t = Trainer_VD(args, loader, model, loss, chkp)
while not t.terminate():
    t.train()
    t.test()
chkp.done()
