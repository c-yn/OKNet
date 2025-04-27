# Omni-Kernel Network for Image Restoration [AAAI-24]


<!--[![](https://img.shields.io/badge/OKNet-paper-blue.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/27907)    -->

<!--[![](https://img.shields.io/badge/OKNet-Appendix-blue.svg)](https://drive.google.com/file/d/1IVXjApjDWJiLc434d70cOhBpNk9UsyFp/view?usp=sharing)-->





## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~

Please download pillow package using Conda instead of pip.



ITS FLOPs: 39.67G, Params: 4.72M
Training and testing details can be found in the individual directories.

## [Models](https://drive.google.com/drive/folders/1jrqqTBFHi2XNvBb9-rm9n0fZHfYZabFw?usp=sharing)
## [Images](https://drive.google.com/drive/folders/1FuaHw5Wr9PTSKAKn2qEuE-IN1Ye-O8Xj?usp=sharing)



## Citation
~~~
@inproceedings{cui2024omni,
  title={Omni-Kernel Network for Image Restoration},
  author={Cui, Yuning and Ren, Wenqi and Knoll, Alois},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={1426--1434},
  year={2024}
}
~~~


## Contact
Should you have any question, please contact Yuning Cui.
