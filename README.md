## Deep LTMLE: Longitudinal Targeted Minimum Loss-based Estimation with Temporal-Difference Heterogeneous Transformer (ICML 2024)

[![Python 3.9+](https://img.shields.io/badge/Platform-Python%203.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.3.0](https://img.shields.io/badge/Implementation-Pytorch-brightgreen.svg)](https://pytorch.org/)

Toru Shirakawa, Yi Li, Yulun Wu, Sky Qiu, Yuxuan Li, Mingduo Zhao, Hiroyasu Iso, Mark van der Laan

**[Paper](https://arxiv.org/abs/2404.04399)**
**[Package](https://github.com/shirakawatoru/dltmle)**

Deep LTMLE estimates the mean counterfactual outcome under a dynamic treatment policy and its confidence interval using the iterated conditional expectation (ICE) with targeting 🎯

## Acknowledgement

The code includes some parts from [DeepACE](https://github.com/dennisfrauen/deepace).

## Reproduce Experiments

```bash
python -m venv .venv -r requirements.txt
source .venv/bin/activate
source scripts/run-all.sh
```

## Citation

```BiBTeX
@article{shirakawa_deepltmle_2024,
      title={Longitudinal Targeted Minimum Loss-based Estimation with Temporal-Difference Heterogeneous Transformer}, 
      author={Toru Shirakawa and Yi Li and Yulun Wu and Sky Qiu and Yuxuan Li and Mingduo Zhao and Hiroyasu Iso and Mark van der Laan},
      year={2024},
      journal={International Conference of Machine Learning}
}
```

