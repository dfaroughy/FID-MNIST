# Fréchet Classifier Distance for MNIST

To effectively evaluate generative models on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (or similar single-channel $28\times28$ pixel images), 
we propose a metric similar to the [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500). This metric replaces the `InceptionNet-v3` 
architecture with a classifier better suited for MNIST-like images.

While more complex architectures could be used, our experiments demonstrate that a slightly modified
[`LeNet-5` classifier](http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf), achieving classification accuracy above $95$% for MNIST digits, 
leads to a good metric. The resulting metric, termed the *Fréchet LeNet Distance* (FLD), is computed from the 
Fréchet distance between Gaussian feature vectors extracted from the middle connected layer (FLD$_2$) of the LeNet 
classifier, assessed on both target and reference datasets.

We show that the FLD$_2$ metric correlates well with human judgment when applied to the MNIST digits dataset. The FLD value increases monotonically 
as more corruption is introduced to the images. This trend is consistent across various distortions, such as:

- Gaussian noise, 
- Gaussian blur, 
- blackout,
- swirl effects,
- pixelization,
- rectangular cropping.

![Fréchet Distance Distortions](/plots/Noise_combined_plot.png)
![Fréchet Distance Distortions](/plots/Blur_combined_plot.png)
![Fréchet Distance Distortions](/plots/Blackout_combined_plot.png)
![Fréchet Distance Distortions](/plots/Swirl_combined_plot.png)
![Fréchet Distance Distortions](/plots/Pixelize_combined_plot.png)
![Fréchet Distance Distortions](/plots/Crop_combined_plot.png)
