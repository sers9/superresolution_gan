# Superresolution using GAN
PyTorch 0.4.1

Training of a typical GAN architecture for increasing the resolution of downsampled images. For this purpose, training split of the [Tiny-ImageNet dataset](https://tiny-imagenet.herokuapp.com/) is used. 

Images are firstly downsampled to 32x32 images and fed into a Genarator that tries to reconstruct 64x64 original images. Generator is a U-net-like generator w/o skip connections and Discriminator is an LSGAN  discriminator.
