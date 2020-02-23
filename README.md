# NCSN-TF2.0
Reproduction of "Generative Modeling by Estimating Gradients of the Data Distribution" by Yang Song and Stefano Ermon (NeurIPS 2019) in Tensorflow 2.0.

[Github] | [Paper]

Created for the [Reproducibility Challenge @ NeurIPS 2019].

**Instructions for running the code**

The main file to run is `main.py`. Different options are available depending on the task to perform.

`--experiment train` for training the model.

`--experiment generate` to sample through Langevin dynamics with a trained model.

`--experiment inpainting` for performing inpainting (different patterns of occlusion are available).

`--experiment toytrain` to run the toy experiment.

`--experiment evaluation` for computing inception and FID score.

`--experiment k_nearest` to samples images and finding the k pixel-wise nearest images in the dataset. 

`--experiment intermediate` to sample images and save them at each level of noise.

`--experiment celeb_a_statistics` for computing inception and FID score on CelebA.

`--help` for additional info, or check out `utils.py`.

[Paper]: https://arxiv.org/pdf/1907.05600.pdf
[Github]: https://github.com/ermongroup/ncsn
[Reproducibility Challenge @ NeurIPS 2019]: https://reproducibility-challenge.github.io/neurips2019/
