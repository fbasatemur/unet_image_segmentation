# Pytorch - Unet: for Biomedical Image Segmentation 
For biomedical images, unet was first mentioned in article [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015). In this repository, neuronal membrane segmentation will be perform use the unet architecture in below.

![unet_architecture](https://github.com/fbasatemur/unet_image_segmentation/blob/main/docs/u-net-architecture.png)

If you want more information about unet, you should to examine this: [Freiburg University - Vision](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## Preprocessing
Border reflect is applied to the input images (512x512) to convert the images to 572 x 572 dimensions. Target images were cropped from the center at 388 x 388.


***512x512 - train/0.tiff***             |  ***572x572 border reflect***         | ***388x388 ground truth***
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/fbasatemur/unet_image_segmentation/blob/main/docs/train_0.jpg)  |  ![](https://github.com/fbasatemur/unet_image_segmentation/blob/main/docs/train_0-reflect.png)  |  ![](https://github.com/fbasatemur/unet_image_segmentation/blob/main/docs/train_0-gtruth.png)

## Train - Test
When test/0.tiff is given to 300 Epoch trained model inputs, the following result is obtained.

***512x512 - test/0.tiff***             |  ***388x388 model output***   
:-------------------------:|:-------------------------:
![](https://github.com/fbasatemur/unet_image_segmentation/blob/main/docs/test_0.jpg)  |  ![](https://github.com/fbasatemur/unet_image_segmentation/blob/main/docs/test_0-predict.png) 
