# Run

**Train**
```
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```
