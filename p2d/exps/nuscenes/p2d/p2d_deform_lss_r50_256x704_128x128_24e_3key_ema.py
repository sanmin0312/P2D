# Copyright (c) Megvii Inc. All rights reserved.
"""

"""
from p2d.exps.base_cli import run_cli
from p2d.exps.nuscenes.p2d.p2d_deform_lss_r50_256x704_128x128_24e_3key import \
    P2DLightningModel  # noqa

if __name__ == '__main__':
    run_cli(P2DLightningModel,
            'p2d_deform_lss_r50_256x704_128x128_24e_3key_ema',
            use_ema=True)
