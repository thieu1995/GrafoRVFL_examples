# GrafoRVFL Experiments

### Setup environment

```bash
pip install -r requirements.txt
```

### Run the scripts

```bash
python 01_breast_cancer.py
....
```

### Models

```python
from mealpy import BBO, DE, SHADE, LCO, INFO, SCA, SHIO, EFO, EO, RIME, ARO, HHO, PSO

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
```


```code
1. Biogeography-Based Optimization (BBO)
  Simon, D., 2008. Biogeography-based optimization. IEEE transactions on evolutionary computation, 12(6), pp.702-713.
2. Self-Adaptive Differential Evolution (SADE)
  Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for
    numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE.
3. Success-History Adaptation Differential Evolution (OriginalSHADE)
    Tanabe, R. and Fukunaga, A., 2013, June. Success-history based parameter adaptation for
    differential evolution. In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
4. Life Choice-based Optimization (LCO)
  Khatri, A., Gaba, A., Rana, K.P.S. and Kumar, V., 2020. A novel life choice-based optimizer. Soft Computing, 24(12), pp.9121-9141.
5. weIghted meaN oF vectOrs (INFO)
  Ahmadianfar, I., Heidari, A. A., Noshadian, S., Chen, H., & Gandomi, A. H. (2022). INFO: An efficient optimization
    algorithm based on weighted mean of vectors. Expert Systems with Applications, 195, 116516.
6. QLE Sine Cosine Algorithm (QLE-SCA)
  Hamad, Q. S., Samma, H., Suandi, S. A., & Mohamad-Saleh, J. (2022). Q-learning embedded sine cosine
    algorithm (QLESCA). Expert Systems with Applications, 193, 116417.
7. Success History Intelligent Optimizer (SHIO)
   Fakhouri, H. N., Hamad, F., & Alawamrah, A. (2022). Success history intelligent optimizer. The Journal of Supercomputing, 1-42.
8. Electromagnetic Field Optimization (EFO)
  Abedinpourshotorban, H., Shamsuddin, S.M., Beheshti, Z. and Jawawi, D.N., 2016.
    Electromagnetic field optimization: a physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation, 26, pp.8-22.
9. Adaptive Equilibrium Optimization (AEO)
  Wunnava, A., Naik, M.K., Panda, R., Jena, B. and Abraham, A., 2020. A novel interdependence based
    multilevel thresholding technique using adaptive equilibrium optimizer. Engineering Applications of
    Artificial Intelligence, 94, p.103836.
10. physical phenomenon of RIME-ice  (RIME)
    Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). RIME: A physics-based optimization. Neurocomputing.
11. Lévy flight, and the selective opposition version of the artificial rabbit algorithm (LARO)
  Wang, Y., Huang, L., Zhong, J., & Hu, G. (2022). LARO: Opposition-based learning boosted
    artificial rabbits-inspired optimization algorithm with Lévy flight. Symmetry, 14(11), 2282.
12. Harris Hawks Optimization (HHO)
  Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. and Chen, H., 2019.
    Harris hawks optimization: Algorithm and applications. Future generation computer systems, 97, pp.849-872.
13. Adaptive Inertia Weight Particle Swarm Optimization (AIW-PSO)
  Qin, Z., Yu, F., Shi, Z., Wang, Y. (2006). Adaptive Inertia Weight Particle Swarm Optimization. In: Rutkowski, L.,
    Tadeusiewicz, R., Zadeh, L.A., Żurada, J.M. (eds) Artificial Intelligence and Soft Computing – ICAISC 2006. ICAISC 2006.
    Lecture Notes in Computer Science(), vol 4029. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11785231_48
14. Comprehensive Learning Particle Swarm Optimization (CL-PSO)
  Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer
    for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
```


### Dataset information

Breast_cancer Dataset:
  Samples: 569
  Training data shape: (455, 30)
  Test data shape: (114, 30)
  Training target shape: (455,)
  Test target shape: (114,)
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

Digits Dataset:
  Samples: 1797
  Training data shape: (1437, 64)
  Test data shape: (360, 64)
  Training target shape: (1437,)
  Test target shape: (360,)
  Unique y values: 10
