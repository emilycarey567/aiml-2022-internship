# TODO

1. Image processing basics:
  * Understanding how images are represented and stored as pixel data
  * Basic image manipulation techniques such as resizing, cropping, and color space conversions
  * Edge detection: using OpenCV or other image processing libraries to detect edges in images

2. Convolutional neural networks (CNNs):
  * Understanding how CNNs work and how they are used for image classification
  * Understanding the concept of convolution and how it is used in CNNs
  * Understanding the different layers in a CNN (e.g. convolutional layers, pooling layers, fully-connected layers)
  * Training and optimizing CNNs using techniques such as backpropagation and gradient descent
  * Learning about the ResNet architecture and how it is used for image classification

3. Semantic segmentation:
  * Understanding the difference between semantic segmentation and other types of image segmentation (e.g. instance segmentation, panoptic segmentation)
  * Understanding how semantic segmentation models work and how they are trained
  * Understanding the different types of architectures and techniques used in semantic segmentation models (e.g. encoder-decoder architectures, skip connections, atrous convolutions)
  * Evaluating the performance of semantic segmentation models and techniques for improving their performance (e.g. data augmentation, model ensembles)

4. Applications of semantic segmentation:
  * Understanding how semantic segmentation is used in a variety of applications such as object detection, scene understanding, and image manipulation
  * Understanding the challenges and limitations of using semantic segmentation in real-world scenarios



# Environment

To open up your new enviroment, do:
```bash
source activate opencvconda
```

To make the enviroment again:

```bash
conda create -n opencvconda
source activate opencvconda
conda install opencv matplotlib numpy
```

## Long version

When you open a new terminal, it should look like this:
```bash
(base) student-10-201-00-218:aiml-2022-internship emilycarey$
```

Note the `(base)`, this tells you that you are in the default conda environment. When we were installing opencv2, for some reason, we couldn't get it to install in this default base env. So we made a new conda environment called `opencvconda`. So to use this environment we need to use:

```bash
source activate opencvconda
```

Then your terminal should look like this:

```bash
(opencvconda) student-10-201-00-218:aiml-2022-internship emilycarey$ 
```

Note the change in conda environment we're in "(base)" -> "(opencvconda)". If you ever want to install more pip packages, you can instead google how to install it using "conda". 