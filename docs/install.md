# Installation
**1. Create a conda virtual environment**
```
conda create --name p2d python=3.8 -y
conda activate p2d
```

**2. Install PyTorch (v1.9.0)**
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Clone P2D**
```
https://github.com/sanmin0312/P2D.git
```

**4. Install mmcv, mmdet and mmseg**
```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1
```

**5. Install mmdet3d**
```
cd P2D
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```

**6. Install requirements**
```
pip install -r requirements.txt
python setup.py develop
```
