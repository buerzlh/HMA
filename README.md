# HMA
PyTorch implementation for **Homeomorphism Alignment for Unsupervised Domain Adaptation** (ICCV2023). This repository is based on framework from [CAN](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation) and modified part of the code. 

The installation can refer to CAN.

Please download datasets to experiments/datasets/ folder.

## Training:
```
CUDA_VISIBLE_DEVICES=X python tools/train.py --method CAN_INN --cfg experiments/config/home/cfgXXXX.yaml
```



## Citation
If you think our paper or code is helpful to you, we very much hope that you can cite our paper, thank you very much.

```
@inproceedings{zhou2023homeomorphism,
  title={Homeomorphism Alignment for Unsupervised Domain Adaptation},
  author={Zhou, Lihua and Ye, Mao and Zhu, Xiatian and Xiao, Siying and Fan, Xu-Qian and Neri, Ferrante},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18699--18710},
  year={2023}
}
```
