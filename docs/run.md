# Run

**Train**
```
python [EXP_PATH] --amp_backend native -b 4 --gpus 4

#python .p2d/exps/nuscenes/p2d/p2d_deform_lss_r50_256x704_128x128_24e_3key.py --amp_backend native - b 4 --gpus 4
```

**Evaluation**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4

#python .p2d/exps/nuscenes/p2d/p2d_deform_lss_r50_256x704_128x128_24e_3key.py --ckpt_path ./outputs/p2d_deform_lss_r50_256x704_128x128_24e_3key/lightning_logs/version_0/23.pth -e - b 4 --gpus 4
```
