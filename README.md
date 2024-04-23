# Fréchet Classifier Distance for MNIST

To effectively evaluate generative models on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (or similar single-channel $28\times28$ pixel images), 
we propose a metric similar to the [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500). This metric replaces the `InceptionNet-v3` 
architecture with a classifier better suited for MNIST-like images.

While more complex architectures could be used, our experiments demonstrate that a slightly modified
[`LeNet`](http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf) classifier, achieving an accuracy above $95$% for MNIST digit classification, 
leads to a good similarity metric. The *Fréchet LeNet Distance* (FLD) is the Frechet distance between the target and reference comnputed computed from feature of the last layers of a pre-trained `LeNet-5` 
classifier.

We show that the $FLD_2$ metric (computed from the features in the second fully-connected layer) correlates well with human judgment when applied to the MNIST digits dataset. Similarily, $FLD_3$ is a good metric for a binerized the MNIST dataset. The resulting FLD values increases monotonically as more corruption is introduced to the images. This trend is consistent across various distortions, such as:

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
