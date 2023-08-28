# Copyright (c) Megvii Inc. All rights reserved.

from p2d.exps.base_cli import run_cli
from p2d.exps.nuscenes.base_exp import \
    P2DLightningModel as BaseP2DLightningModel
from p2d.models.p2d import P2D

import copy
import torch

class P2DLightningModel(BaseP2DLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-2, -4]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * 2
        self.head_conf['bev_neck_conf']['in_channels'] = [80 * 2, 160, 320, 640]

        self.prediction_conf = copy.deepcopy(self.head_conf)
        self.prediction_conf['bev_backbone_conf']['in_channels'] = 80 * (len(self.key_idxes))
        self.prediction_conf['bev_neck_conf']['in_channels'] = [80 * (len(self.key_idxes)), 160, 320, 640]
        self.temporalfusion_config = dict(
                                        embed_dims=80,
                                        num_heads=8,
                                        num_levels=len(self.key_idxes)+1,
                                        num_layers=6,
                                        num_points=9,
                                        num_bev_queue=2,
                                        im2col_step=64,
                                        dropout=0.1,
                                        positional_encoding=dict(
                                            type='LearnedPositionalEncoding',
                                            num_feats=80 // 2,
                                            row_num_embed=128,
                                            col_num_embed=128,
                                        ),
                                        batch_first=True,
                                        ffn_cfg = dict(
                                            type='FFN',
                                            embed_dims=80,
                                            feedforward_channels=80,
                                            num_fcs=2,
                                            ffn_drop=0.1,
                                            act_cfg=dict(type='ReLU', inplace=True),
                                        ),
                                        norm_cfg = dict(type='LN')
                                        )

        self.model = P2D(self.backbone_conf,
                         self.head_conf,
                         self.prediction_conf,
                         self.temporalfusion_config,
                         is_train_depth=True)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, past_preds, depth_preds = self(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
            prediction_loss = self.model.module.loss(targets, past_preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
            prediction_loss = self.model.loss(targets, past_preds)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        self.log('detection_loss', detection_loss)
        self.log('prediction_loss', prediction_loss)
        self.log('depth_loss', depth_loss)
        return detection_loss + depth_loss + prediction_loss * 0.5


if __name__ == '__main__':
    run_cli(P2DLightningModel,
            'p2d_deform_lss_r50_256x704_128x128_24e_3key')
