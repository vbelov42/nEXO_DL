# nEXO_DL
This is the package developed for background and 0vbb event separation in the next-generation Entrich Xenon Observatory (nEXO). The package is built on pytorch platform. It takes the nEXO charge simulation as input (possibly adding photon information in the future), and utilizes a 18-layer ResNet for classification.

## Content of the package
 1. nEXO2DChargeImage.py - script to convert nEXO charge simulation to two images. Only two channels of the image are currently used. The third channel is open for future addition of photon information.
 2. image2dcharge_csv.py - script to build csv file for dataset build.
 3. nEXO_DL.py - main script for deep learning model construction, training, and testing.
 4. resnet_example.py - ResNet configuration file. copied from https://github.com/DeepLearnPhysics/pytorch-resnet-example
 
## Requirements
 * pytorch
 * ROOT6
 * dataset - The nEXO simulation and simulation result is not open. 
 
