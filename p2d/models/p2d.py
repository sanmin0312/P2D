from torch import nn
import torch

from p2d.layers.backbones.base_lss_fpn_p2d import BaseLSSFPN_P2D
from p2d.layers.heads.bev_depth_head import BEVDepthHead
from p2d.layers.neck.temporal_fusion import TemporalFusion

__all__ = ['P2D']


class P2D(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, prediction_conf, temporalfusion_config, is_train_depth=False):
        super(P2D, self).__init__()
        self.backbone = BaseLSSFPN_P2D(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.prediction = BEVDepthHead(**prediction_conf)
        self.tempfusion = TemporalFusion(**temporalfusion_config)

        self.is_train_depth = is_train_depth

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            bs, frames, ch, bevx, bevy = x.shape
            x_past = x[:,1:]
            preds = self.prediction(x_past.reshape(bs, -1, bevx, bevy))
            out = self.tempfusion(x, preds)
            dets = self.head(torch.cat((out, x[:,0]), 1))

            return dets, preds, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            bs, frames, ch, bevx, bevy = x.shape
            x_past = x[:,1:]
            preds = self.prediction(x_past.reshape(bs, -1, bevx, bevy))
            out = self.tempfusion(x, preds)
            dets = self.head(torch.cat((out, x[:,0]), 1))
            return dets


    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)