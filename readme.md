## Masked Spiking Transformer (ICCV-2023)

Masked Spiking Transformer, ICCV'23: [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Masked_Spiking_Transformer_ICCV_2023_paper.html). [[Poster]](https://github.com/bic-L/Masked-Spiking-Transformer/files/12871675/Masked_Spiking_Transforer_Poster.pdf). [[Video]](https://user-images.githubusercontent.com/141820457/274327624-16db42f2-2df1-4127-bbac-f746b5aa9534.mp4)  [[Follow-up work with detailed energy cost analysis]](https://www.researchsquare.com/article/rs-6004117/v1)

### Abstract
The combination of Spiking Neural Networks (SNNs) and Transformers has attracted significant attention due to their potential for high energy efficiency and high-performance nature. However, **there still remains a considerable challenge to achieving performance comparable to artificial neural networks in real-world applications** .
Besides, **the mainstream method for ANN-to-SNN conversion trade increased simulation time and power for high-performance SNN, and generally consumes lots of computational resources when making inference**.

To address this issue, we propose an energy-efficient architecture, **the Masked Spiking Transformer, that combines the benefits of SNNs and the high-performance self-attention mechanism in Transformer utilizing the ANN-to-SNN conversion methods**. Furthermomre, our method, called R**andom Spike Masking, prunes input spikes during both training and inference to reduce SNN computational costs**. 

The Masked Spiking Transformer combines the self-attention mechanism and the ANN-to-SNN conversion method, achieving state-of-the-art accuracy on both static and neuromorphic datasets. Experimental results demonstrate the RSM method reduces redundant spike operations while keeping model performance over a certain range of mask rates across various model architectures. For instance, the RSM method reduces MST model power by 26.8% at a 75% mask rate with no performance drop. 

<div align="center"> <img src="https://github.com/bic-L/Masked-Spiking-Transformer/blob/master/figures/acc.jpg" width="700" height="500"  alt="acc"/> </div>
<img src="https://github.com/bic-L/Masked-Spiking-Transformer/blob/master/figures/main.jpg"  alt="acc"/><br/>

### Running the Code

Checkpoints:

MST( 0% masking rate): [Cifar-10](https://github.com/bic-L/Masked-Spiking-Transformer/releases/download/checkpoint/Cifar10_checkpoint.pth), [Cifar-100](https://github.com/bic-L/Masked-Spiking-Transformer/releases/download/checkpoint/Cifar100_checkpoint.pth), [Imagenet](https://github.com/bic-L/Masked-Spiking-Transformer/releases/download/checkpoint/imagenet_checkpoint.pth)

MST( 75% masking rate): [Cifar-10](https://github.com/bic-L/Masked-Spiking-Transformer/releases/download/checkpoint_with_mask/cifar10_75._masking_ratio_checkpoint.pth), [Cifar-100](https://github.com/bic-L/Masked-Spiking-Transformer/releases/download/checkpoint_with_mask/Cifar100_75._masking_ratio_checkpoint.pth), [Imagenet](https://github.com/bic-L/Masked-Spiking-Transformer/releases/download/checkpoint_with_mask/imagenet_75._masking_ratio_checkpoint.pth)

For more training details, please check out our paper and supplementary material. (Note: we used 8×3090 GPU cards for training)

#### 1. Pre-training ANN MST with QCFS function on ImageNet with multiple GPUs:
```bash
torchrun --nproc_per_node 8 main.py --cfg configs/mst/MST.yaml --batch-size 128 --masking_ratio masking_rate
```

#### 2. SNN Validation:
```bash
torchrun --nproc_per_node 8 main.py --cfg configs/mst/MST.yaml --batch-size 128 --snnvalidate True --sim_len 128 --pretrained /path/to/weight/ --dataset imagenet --masking_ratio masking_rate
```
- `--sim_len`: timestep of SNN.
- `--snnvalidate`: enalbes SNN validation.
- `--dataset`: name of dataset, choice=['imagenet', 'Cifar100', 'Cifar10'].

