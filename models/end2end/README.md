### End2end 

* The original repository is available here: https://github.com/lishen/end2end-all-conv 
* The model operates on a single input mammogram.
* The model consists only of convolutional layers as shown in the model architecture below.

![end2end model](end2end_model.jpg)

* There are 4 different hyper parameters in the config.txt file:
    * `MEAN_PIXEL_INTENSITY` - this is the average pixel intensity of the images in the train set after they are rescaled to have a max value of 255. This is subtracted from the images as part of the preprocessing. For example, if evaluating on the DDSM test set, then `MEAN_PIXEL_INTENSITY=52.18`. If evaluating on the INbreast test set, then `MEAN_PIXEL_INTENSITY=44.4`. For NYU datasets, the value is 31.28; for CMMD dataset, it's 18.01. If you are using a custom data test set, then you will need to compute this value.
    * `MODEL` - there are 6 different models available, each with a different architecture. Please refer to the original repository to see the differences between each model. You can select from the following:
        * ddsm_resnet50_s10_[512-512-1024]x2.h5 
        * ddsm_vgg16_s10_512x1.h5
        * ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5
        * ddsm_YaroslavNet_s10.h5
        * inbreast_vgg16_512x1.h5
        * inbreast_vgg16_[512-512-1024]x2_hybrid.h5
    * `NUM_PROCESSES` - controls how many processes are used in the preprocessing of the images.
    * `PREPROCESS_FLAG` - If this is set to True, then the images are always preprocessed even if there already exist preprocessed versions.


