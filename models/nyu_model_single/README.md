### NYU Multiview Model - Single 
* This is a special instance of the NYU Multiview Model that is able to operate on single images.
* It has the same hyper parameters as NYU Multiview Model
    * `HEATMAP_BATCH_SIZE` - controls the minibatch size used when generating heatmaps.
    * `NUM_EPOCHS` - controls the number of epochs over validation set (with different preprocessing seeds) to be averaged in the output of the classifiers.
    * `USE_HEATMAPS` - controls whether to use heatmaps or not. If using heatmaps, the process consists of two steps: first a deep network is ran on all 256x256 patches from the main image to generate two heatmaps (one for malignancy, one for benignity). Then, a shallower network is used on the full resolution concatenation of the input image and two heatmaps. If not using heatmaps, the shallower network is used on the input images only.