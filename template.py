#!/usr/bin/env python
# -*- coding: utf-8 -*-

def set_template(args):
    args.data_train = 'BLUR_VIDEO'
    args.dir_data = './train_dataset'
    args.data_test = 'BLUR_VIDEO'
    args.dir_data_test = './train_dataset'
    args.model = "deblur"

    # args.test_only = True
    # args.pre_train = '/home/weihao/weihao/deblur/Video_Deblur_edit/experiment/model/model.pt'

    if args.model == 'deblur':

        args.flowModel_path = './pretrained_model/flownets_EPE1.951.pth.tar'
        args.loss = "1*L1"
        args.save = "deblur"
        if args.test_only:
            args.save = "deblur_test"
