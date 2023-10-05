## Masked Spiking Transformer (ICCV2023)

### *checkpoints will be released soon!*

### Abstract
The combination of Spiking Neural Networks (SNNs) and Transformers has attracted significant attention due to their potential for high energy efficiency and high-performance nature. However, existing works on this topic typically rely on direct training, which can lead to suboptimal performance. To address this issue, we propose to leverage the benefits of the ANN-to-SNN conversion method to combine SNNs and Transformers, resulting in significantly improved performance over existing state-of-the-art SNN models. Furthermore, inspired by the quantal synaptic failures observed in the nervous system, which reduce the number of spikes transmitted across synapses, we introduce a novel Masked Spiking Transformer (MST) framework. This incorporates a Random Spike Masking (RSM) method to prune redundant spikes and reduce energy consumption without sacrificing performance. Our experimental results demonstrate that the proposed MST model achieves a significant reduction of 26.8% in power consumption when the masking ratio is 75% while maintaining the same level of performance as the unmasked model.

![Main Figure](figures/main.jpg)

![performance](figures/acc.jpg)

The paper is available at [arxiv](https://arxiv.org/pdf/2210.01208.pdf). 
### Running the Code

#### 1. Pre-training ANN MST with QCFS function on ImageNet with multiple GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/mst/MST.yaml --batch-size 128
```

#### 2. SNN Validation:
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/mst/MST.yaml --batch-size 128 --snnvalidate True --sim_len 128
```
- `--sim_len`: timestep of SNN.
- `--snnvalidate`: enalbes SNN validation.
