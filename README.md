# Domain-Guided Conditional Diffusion Model for Unsupervised Domain Adaptation (Accepted by Neural Networks)

Official PyTorch implementation of Domain-Guided Conditional Diffusion Model for Unsupervised Domain Adaptation

Yulong Zhang*, Shuhao Chen*, Weisen Jiang, Yu Zhang, Jiangang Lu, James T. Kwok.


## Abatract
Limited transferability hinders the performance of a well-trained deep learning model when applied to new application scenarios. Recently, Unsupervised Domain Adaptation (UDA) has achieved significant progress in addressing this issue via learning domain-invariant features. However, the performance of existing UDA methods is constrained by the possibly large domain shift and limited target domain data. To alleviate these issues, we propose a Domain-guided Conditional Diffusion Model (DCDM), which generates high-fidelity target domain samples, making the transfer from source domain to target domain easier. DCDM introduces class information to control labels of the generated samples, and a domain classifier to guide the generated samples towards the target domain. Extensive experiments on various benchmarks demonstrate that DCDM brings a large performance improvement to UDA.
<!-- 
## Installation

setup.py
``` -->

## Usage
You can find scripts in the directory scripts. 
The code for UDA method: [MCC](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/image_classification), [ELS](https://github.com/yfzhang114/Environment-Label-Smoothing), [SSRT](https://github.com/tsun/SSRT).

## Contact
If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 
- Yulong Zhang (zhangylcse@zju.edu.cn)

or describe it in Issues.


## Acknowledgement

Our implementation is based on the [ED-DPM](https://github.com/ZGCTroy/ED-DPM), [Guided-diffusion](https://github.com/openai/guided-diffusion), [dpm-solver](https://github.com/LuChengTHU/dpm-solver).

## Citation
If you find our paper or codebase useful, please consider citing us as:
```latex
@article{zhang2023domain,
  title={Domain-guided conditional diffusion model for unsupervised domain adaptation},
  author={Zhang, Yulong and Chen, Shuhao and Jiang, Weisen and Zhang, Yu and Lu, Jiangang and Kwok, James T},
  journal={arXiv preprint arXiv:2309.14360},
  year={2023}
}