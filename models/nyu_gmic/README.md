### NYU GMIC 

* The original repository is available here: https://github.com/nyukat/GMIC 
* The NYU Global Multiple Instance Classifier operates on a single input mammogram and generates two predictions, probability of malignant lesion in the breast and probability of benign lesion in the breast.
* The architecture of the model is shown below. The model combines global information from the entire image along with local information extracted from interesting patches in the image to make a prediction.

![gmic architecture](nyu_gmic.png)

* There are 3 different hyper parameters in the config.txt file
    * `MODEL_INDEX` - there are 5 different classifiers available. Use values 1-5 to select a specific classifier. Use 'ensemble' to classify using an ensemble of all 5 classifiers. The classifiers differ with respect to the value of percent_t parameter (from 0.02 with model_index '1' to 0.1 with model_index '5').
    * `NUM_PROCESSES` - controls how many processes are used in the preprocessing of the images
    * `PREPROCESS_FLAG` - if set to True, forces the model to redo preprocessing even if it has already been completed. Otherwise, will use the preexisting preprocessed images.
