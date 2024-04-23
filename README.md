# Fréchet Classifier Distance for MNIST

To effectively evaluate generative models on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (or similar single-channel $28\times28$ pixel images), 
we propose a metric similar to the [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500). This metric replaces the `InceptionNet-v3` 
architecture with a classifier better suited for MNIST-like images.

Our experiments demonstrate that a slightly modified
[`LeNet-5`](http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf) architecture, with classification accuracy above $99$% for MNIST digits, 
leads to a good similarity metric that correlates well with human judgement. We define the *Fréchet LeNet Distance* (FLD) as the Frechet distance between hidden features of the last layers of a pre-trained LeNet classifier evaluated on target and reference datasets.

As can be seen below, the FLD values of the three fully-connected layers (third is the output before softmaxing) increases monotonically as more corruption is introduced to the images. This trend is consistent across various distortions:

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

From here one sees that a good metric for MNIST digits is the averaged features over the first two fully connected layers:

$$
FLD(\rm MNIST) = \frac{FLD_1+FLD_2}{2}
$$

