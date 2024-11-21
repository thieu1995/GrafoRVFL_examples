# GrafoRVFL Experiments

### Setup environment

```bash
pip install -r requirements.txt
```

### Run the scripts

```bash
python 01_iris.py
....
```

LIST_MODELS = [
    {"name": "BBO-RVFL", "class": "OriginalBBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SADE-RVFL", "class": "SADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SHADE-RVFL", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "LCO-RVFL", "class": "OriginalLCO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "INFO-RVFL", "class": "OriginalINFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "QLE-SCA-RVFL", "class": "QleSCA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SHIO-SCA-RVFL", "class": "OriginalSHIO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "EFO-RVFL", "class": "OriginalEFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "A-EO-RVFL", "class": "AdaptiveEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "RIME-RVFL", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "LARO-RVFL", "class": "LARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "HHO-RVFL", "class": "OriginalHHO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "AIW-PSO-RVFL", "class": "AIW_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "CL-PSO-RVFL", "class": "CL_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
]



### Dataset information

Iris Dataset:
  Samples: 150
  Training data shape: (120, 4)
  Test data shape: (30, 4)
  Training target shape: (120,)
  Test target shape: (30,)
  Unique y values: 3

Breast_cancer Dataset:
  Samples: 569
  Training data shape: (455, 30)
  Test data shape: (114, 30)
  Training target shape: (455,)
  Test target shape: (114,)
  Unique y values: 2

Digits Dataset:
  Samples: 1797
  Training data shape: (1437, 64)
  Test data shape: (360, 64)
  Training target shape: (1437,)
  Test target shape: (360,)
  Unique y values: 10

Wine Dataset:
  Samples: 178
  Training data shape: (142, 13)
  Test data shape: (36, 13)
  Training target shape: (142,)
  Test target shape: (36,)
  Unique y values: 3

Phoneme Dataset:
  Samples: 5404
  Training data shape: (4323, 5)
  Test data shape: (1081, 5)
  Training target shape: (4323,)
  Test target shape: (1081,)
  Unique y values: 2

Waveform Dataset:
  Samples: 5000
  Training data shape: (4000, 40)
  Test data shape: (1000, 40)
  Training target shape: (4000,)
  Test target shape: (1000,)
  Unique y values: 3

Magic_gamma Dataset:
  Samples: 13376
  Training data shape: (10700, 10)
  Test data shape: (2676, 10)
  Training target shape: (10700,)
  Test target shape: (2676,)
  Unique y values: 2

Diabetes Dataset:
  Samples: 768
  Training data shape: (614, 8)
  Test data shape: (154, 8)
  Training target shape: (614,)
  Test target shape: (154,)
  Unique y values: 2

Boston Dataset:
  Samples: 506
  Training data shape: (404, 22)
  Test data shape: (102, 22)
  Training target shape: (404,)
  Test target shape: (102,)
  Unique y values: 207

California Dataset:
  Samples: 20640
  Training data shape: (16512, 8)
  Test data shape: (4128, 8)
  Training target shape: (16512,)
  Test target shape: (4128,)
  Unique y values: 3675
