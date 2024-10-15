# GTSM: graph transformer-based surrogate model for learning physical systems

The work of this article is based on the work of Professor Tailin Wu of Westlake University. Please refer to Professor Tailin Wu's work （[github](https://github.com/snap-stanford/lamp/tree/master) ）for more details

based on work of Pro. Wu, Wr have made some attempts in solid mechanics. Initially we wanted to apply Tailin Wu's work to multi-resolution scenarios in solid mechanics, such as fracture mechanics. However, during the implementation, we found that it was beyond our capabilities due to the large amount of GPU memory required. In the end, we turned to some simple solid mechanics cases.

## Note
This repository contains training data for 3D cantilever beams, 1D nonlinear equations and paper folding simulations. Please refer to Wu's work.
The dataset files can be downloaded via this [link](https://drive.google.com/drive/my-drive).
 download the files under "tetramesh_beam_data/" in the link into the "data/tetramesh_beam_data/" folder in the local repo.
 If have any questions, please contact me: fengbo19940401@126.com

## Some improvements in this work

1.The graph transformer with attention mechanism is proposed to aggregate information from neighboring nodes.

We believe that adjacent points in a physical system have similar properties, so we limit the attention mechanism to the neighboring nodes of the target nodes in combination with graph data, and do not perform a global attention mechanism.

2.A multi-step prediction strategy is employed in the loss function to ensure robust long-term prediction capability.

Recent work has found that single-step prediction can effectively avoid the oscillation problem during training.

3.Innovative symlog and symexp functions are utlized to predict stress in solid mechanics.

## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2023learning,
    title={Learning Controllable Adaptive Simulation for Multi-resolution Physics},
    author={Tailin Wu and Takashi Maruyama and Qingqing Zhao and Gordon Wetzstein and Jure Leskovec},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=PbfgkZ2HdbE}
}
@article{FENG2024117410,
    title = {The novel graph transformer-based surrogate model for learning physical systems},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {432},
    pages = {117410},
    year = {2024},
    issn = {0045-7825},
    doi = {https://doi.org/10.1016/j.cma.2024.117410},
    url = {https://www.sciencedirect.com/science/article/pii/S0045782524006650},
    author = {Bo Feng and Xiao-Ping Zhou}
}
```
