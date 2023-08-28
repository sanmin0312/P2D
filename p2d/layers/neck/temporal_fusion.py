import torch
import torch.nn as nn
from p2d.layers.modules.temporal_self_attention import TemporalCrossAttentionLayer, TemporalDeformableAttention
from mmcv.cnn.bricks.transformer import build_positional_encoding
from p2d.layers.backbones.base_lss_fpn_p2d import Mlp

__all__ = ['TemporalFusion']


class TemporalFusion(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_levels,
                 num_layers,
                 num_points,
                 num_bev_queue,
                 im2col_step,
                 dropout,
                 batch_first,
                 positional_encoding,
                 ffn_cfg,
                 norm_cfg,
                 ):
        super(TemporalFusion, self).__init__()

        self.querymlp = Mlp(70, embed_dims, embed_dims)
        self.time_encoding = nn.Embedding(num_levels, embed_dims)
        temporalatt = TemporalCrossAttentionLayer(embed_dims=embed_dims,
                                                  num_heads=num_heads,
                                                  num_levels=num_levels,
                                                  num_points=num_points,
                                                  num_bev_queue=num_bev_queue,
                                                  im2col_step=im2col_step,
                                                  dropout=dropout,
                                                  batch_first=True,
                                                  ffn_cfg = ffn_cfg,
                                                  norm_cfg = norm_cfg
                                                  )
        self.deformatt = TemporalDeformableAttention(temporalatt, num_layers=num_layers, embed_dims=embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)

    def forward(self,
                x,
                preds,
                ):
        bs, frames, ch, bevx, bevy = x.shape

        query, mask_ind = self.preds2query(preds, bs)

        spatial_shapes = torch.tensor([[bevx, bevy]]).repeat(frames, 1)
        spatial_shapes = spatial_shapes.type(torch.LongTensor).to(query.device)

        time_pos = self.time_encoding.weight
        value = x.permute(0, 1, 3, 4, 2).contiguous()
        value = value + time_pos[None, :, None, None, :]
        value = value.reshape(bs, -1, ch)

        query_mask = torch.zeros((bs, spatial_shapes[0,0], spatial_shapes[0,1]),
                               device=query.device).to(query.dtype)
        query_pos = self.positional_encoding(query_mask).to(query.dtype)
        query_pos = query_pos.flatten(2).permute(0, 2, 1)
        query_pos = query_pos.gather(1, mask_ind.repeat(1,1,query_pos.shape[-1]))

        out=self.deformatt(query, value, query_pos, spatial_shapes, mask_ind)
        out = out.permute(0, 2, 1).contiguous()

        mask_ind_fill = mask_ind.repeat(1,1,ch)
        out_fill = torch.zeros((bs, ch, bevx*bevy)).to(out.device)
        out_fill = out_fill.scatter(-1, mask_ind_fill.permute(0,2,1).contiguous(), out)
        out_fill = out_fill.reshape(bs, ch, bevx, bevy)

        return out_fill


    def preds2query(self, preds, bs, k=1024, eps=1e-4):

        query=list()
        heatmaps=list()
        preds = list(preds)
        tasks = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']

        for pred in preds:
            # for task in pred[0]:
            for task in tasks:
                if task == 'heatmap':
                    heatmaps.append(torch.clamp(pred[0][task].sigmoid(), min=eps, max=1 - eps))
                    query.append(pred[0][task])

                else:
                    query.append(pred[0][task])

        heatmaps = torch.cat(heatmaps, 1)
        heatmap_mask = torch.max(heatmaps, 1) # [bs, 1, 128, 128]
        heatmap_mask = heatmap_mask.values.reshape(bs, -1)
        heatmap_mask, mask_ind = heatmap_mask.topk(k=k)
        mask_ind = mask_ind[:, :, None]

        query = torch.cat(query, 1)
        query = query.reshape(bs, query.shape[1], -1)
        query = query.permute(0, 2, 1).contiguous()

        query = query.gather(1, mask_ind.repeat(1, 1, query.shape[-1]))
        query = self.querymlp(query)

        return query, mask_ind