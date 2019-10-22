# NCSN-TF2.0
Reproduction of "Generative Modeling by Estimating Gradients of the Data Distribution" by Yang Song and Stefano Ermon (NeurIPS 2019) in Tensorflow 2.0.

[Github] | [Paper]

How to run: `python train.py --dataset mnist`

Created for the [Reproducibility Challenge @ NeurIPS 2019].


TODO list:
1. [x] Think about other evaluation metrics
2. [x] Compute inception score
4. Speed up inception score computation
3. Modify the name of the saved model according to number of filters and maybe other info
9. [x] CelebA and Cifar-10
5. Evaluate FID score with 1000 images per checkpoint for selecting final best model in CelebA and Cifar-10
3. Properly save the loss while training
6. Baseline
4. [?] Evaluate test loss on the whole test set while training

1. INPAINTING
    8. [x] Inpainting, plotting images
    9. Try different occlusion shapes

1. TOY EXAMPLES
    10. Two gaussians
    11. SSM loss with and without noise

1. ABLATION
    1. Compare our code with what was derivable from the paper.
    7. Ablation study on eps and T
    12. Ablation for the architecture: no dilation, number of RCU, number of filters
    14. Difference in the equations (epsilon and norm)
    15. Different initialization for norm or conv
    16. Different noise levels (not geometric)
1. INNOVATION
    11. Denoising
    13. Investigate about Langevin dynamics and maybe modify sampling

[Paper]: https://arxiv.org/pdf/1907.05600.pdf
[Github]: https://github.com/ermongroup/ncsn
[Reproducibility Challenge @ NeurIPS 2019]: https://reproducibility-challenge.github.io/neurips2019/
