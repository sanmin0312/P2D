# Data Preparation

**1. Downlaod nuScenes official dataset & make symlink**
```
ln -s [nuscenes root] ./data/
```

The directory should be as follows.
```
BEVDepth
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

**2. Prepare infos**
```
python scripts/gen_info.py
```

**3. Generate lidar depth**
```
python scripts/gen_depth_gt.py
```
