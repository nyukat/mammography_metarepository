# NYU Breast Cancer Classification Metarepository
## Introduction
This metarepository is a project aimed to accelerate and standardize research and evaluation in deep learning for screening mammography. It includes ready-to-use Docker images of several screening mammography models. There are two main usecases for the metarepository:

1. Developers of deep learning models can provide their implementations in the form of Docker images. This enables fair comparison with other models on various datasets.
2. Data owners can evaluate multiple state-of-the-art models with very little user involvement and without the need to implement the models or their preprocessing pipelines.

![Overview of metarepository](figs/metarepository-overview.JPG "Overview")

## Prerequisites
 * Docker 19.03.6
 * [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
 * Python 3.6-3.8
    * Cython (0.29.21)
    * scikit-learn (0.24.1)
    * pandas (0.25.2)
    * matplotlib (3.3.3)

## Using the metarepository
### 1. Set up users
To avoid running the Docker container as root and avoid permission issues when accessing files inside the container, we create users and groups in the container that correspond to users and groups on the host machine. This is done with the `users.txt` file. It should include groupname and group_id and the usernames and user_ids of the people belonging to this group who are granted access to run the containers/models. 

*Simple setup: single user*

If it does not matter whether other users will have access to dockerized models, simply fill `users.txt` file with your username and user ID. On Linux, you can get your user ID by running `id -u <your_username>` command. Your `users.txt` file will look like the following (please note *two exact lines*):

```
username,user_id
username,user_id
```

*Access for multiple users*

If you would like for multiple users to run models, they need to belong to the same group. For example, if `user1`, `user2`, and `user3` belong to `group1` and should be able to run the containers, then `users.txt` will look like the following:

```
groupname,group_id
username1,user_id1
username2,user_id2
username3,user_id2
```

When creating Dockerfiles for your own model, we suggest you take a similar approach to avoid running the container with root privileges.

### 2. Run model on sample data

In order to run a model with included sample data, you need to populate the `users.txt` file, and then run the `run.sh` bash script, providing basic arguments:

    bash run.sh <model_name> <experiment_name> <img_path> <prepr_img_path> <label_path> <result_path> <device> <gpu_number>

where the arguments represent:
* `model_name` - Name of the model you want to evaluate. This is equivalent to a directory name in the `/models/` folder. As of now, available models are: `end2end`, `frcnn_cad`, `nyu_glam`, `nyu_gmic`, `nyu_model`, `nyu_model_single`
* `experiment_name` - Name of the experiment; will be used to save results
* `img_path` - path to the directory with input images
* `prepr_img_path` - path to a directory where the preprocessed images created by the model/container will be stored
* `label_path` - path to a pickle (.pkl) that contains the labels, see below for details
* `result_path` - directory where the results (predictions and figures) should be saved
* `device` - Either `gpu` or `cpu`. (Note: `frcnn_cad` model is GPU-only)
* `gpu_number` - Which gpu to use, e.g. `0`, not required if `device==cpu`.

An example command to run a NYU GMIC model on GPU:0 with the sample data included in this repository would look like the following:

    bash run.sh nyu_gmic experiment01 sample_data/images/ sample_data/preprocessed_images/ sample_data/data.pkl predictions/ gpu 0
  
### 3. Changing model parameters
Our models also include an optional configuration file, `config.txt`, which can be found in the model's directory, e.g. `/models/end2end/config.txt` is a configuration file for the end2end model: 

```bash
MEAN_PIXEL_INTENSITY=44.4
MODEL=inbreast_vgg16_[512-512-1024]x2_hybrid.h5
NUM_PROCESSES=10
PREPROCESS_FLAG=True
```

It is specific to the model and contains variables/parameters that can be changed. Please refer to README files in model subdirectories for model-specific details, e.g. `/models/end2end/README.md`.

## How do I use my own data set?
If you are in possession of a data set that you would like to use with the included models, please read the following. There are two parts of a data set that the metarepository expects: *images* and a *data list*.

### Images
*Images* must be saved in a PNG format and stored in a common directory. Models expect 16-bit PNGs. You can use a sample snippet to convert DICOM files to 16-bit PNGs as follows:

```python
import png
import pydicom
def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
                     Set to 16 for 16-bit mammograms, etc.
                     Make sure you are using correct bitdepth!
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    with open(png_filename, 'wb') as f:
        writer = png.Writer(
            height=image.shape[0],
            width=image.shape[1],
            bitdepth=bitdepth,
            greyscale=True
        )
        writer.write(f, image.tolist())
```

If you are converting DICOM images into PNGs, make sure that mammograms are correctly presented after conversion. If there are any VOI LUT functions or Photometric Interpretation conversions necessary, you need to make sure the PNG represents an image after applying those.

### Data list
The *data list* is a pickle file (e.g. data.pkl) containing information for each exam. More specifically, the data list is a pickled list of dictionaries, where each dictionary represents a screening mammography exam. The information in one of these dictionaries is shown below.

```python
{
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
  'cancer_label': {
    'left_malignant': 0, 
    'right_malignant': 0, 
  },
  'horizontal_flip': 'NO',
}
```
* `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` specify the images for each view in the exam. Values of those keys specify file names of the images, e.g. `0_L_CC` means that the model will look for a `0_L_CC.png` file.
* `cancer_label` contains labels for each breast in the exam indicating whether a malignant lesion is present or not (breast-level)
* `horizontal_flip` is used if the model expects all of the images to be facing a certain way. Note that not all of this information is required for all of the models.

**Most importantly, please review the `sample_data` directory. It contains a very simple dataset -- if you follow the convention of the file formats, you should be able to easily convert your dataset to match the requisite format.**

## How do I add my own model?

We strongly encourage contributions and feedback from other research groups. If you want to contribute a model, we ask that you follow these guidelines.

There are three things you should include:
  * Dockerfile - this creates an image that has the environment required to run your model. This should clone any repositories and download any model weights.
  * predict - this is a directory that contains the files necessary to generate predictions
    * predict.sh - a bash script that must be included. See predict.sh in example models.
  * config.txt - a txt file that contains the path that the contents of predict (the directory) will be copied to inside the Docker container as well as any parameters/variables that can be changed. See examples.
  
In addition to these three things, you may also include a help directory, which includes any files necessary for building the Docker image or preprocessing images or any other purposes that the above do not cover.

Your model should work with the sample data provided and `run.sh`. If your model generates image-level predictions, it should include the following columns and be saved as a csv file.

Image-level prediction
image_index  |  malignant_pred  |  malignant_label
-------------|---------------|------------------
0_L-CC       |  0.0081          |  0
0_R-CC       |  0.3259          |  0
0_L-MLO      |  0.0335          |  0
0_R-MLO      |  0.1812          |  0
1_L-CC       |  0.0168          |  0
1_R-CC       |  0.9910          |  1
1_L-MLO      |  0.0139          |  0
1_R-MLO      |  0.9308          |  1
2_L-CC       |  0.0227          |  0
2_R-CC       |  0.0603          |  0
2_L-MLO      |  0.0093          |  0
2_R-MLO      |  0.0231          |  0
3_L-CC       |  0.9326          |  1
3_R-CC       |  0.1603          |  0
3_L-MLO      |  0.7496          |  1
3_R-MLO      |  0.0507          |  0

If you model outputs breast-level predictions, it should include the following columns and be saved as a csv.

Breast-level prediction

| index | left_malignant | right_malignant |
| ----- | -------------- | --------------- |
| 0     | 0.0091         | 0.0179          |
| 1     | 0.0012         | 0.7258          |
| 2     | 0.2325         | 0.1061          |
| 3     | 0.0909         | 0.2579          |


## FAQ
### Which models are currently available in the metarepository?
* Breast Cancer Classifier:
    * breast-level (`nyu_model`)
    * image-level (`nyu_model_single`)
    * [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8861376)
    * [Repository](https://github.com/nyukat/breast_cancer_classifier)

* GMIC (image-level) (`nyu_gmic`)
    * [Paper](https://arxiv.org/pdf/1906.02846.pdf)
    * [Repository](https://github.com/nyukat/GMIC)

* GLAM (image-level) (`nyu_glam`)
    * [Paper](https://openreview.net/pdf?id=nBT8eNF7aXr)
    * [Repository](https://github.com/nyukat/GLAM)

* frcnn_cad (breast-level) (`frcnn_cad`)
    * [Paper](https://www.nature.com/articles/s41598-018-22437-z)
    * [Repository](https://github.com/riblidezso/frcnn_cad)

* end2end-all-conv (breast-level) (`end2end`)
    * [Paper](https://arxiv.org/pdf/1708.09427.pdf)
    * [Repository](https://github.com/lishen/end2end-all-conv)

### What metrics are returned by models?
The following three metrics will be computed and outputted to the terminal:
  * [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
  * [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
  * pr_auc_score
  
In addition to the above metrics, the metarepository will also generate precision-recall curves and ROC curves at both the image- (if applicable) and breast-levels. The locations of these images will be outputted to the terminal along with the metrics.

### What results should be expected on the sample images with the supported models?

Please keep in mind that below results are shown only for reproduction purposes. These are calculated on only 4 exams, therefore have a high variance.

##### Image level:

| Model | ROC AUC  | PR AUC |
| ----------------- | ----- | ----- |
| nyu_glam          | 0.7   | 0.451 |
| nyu_gmic          | 0.867 | 0.851 |
| nyu_model         | -     | -     |
| nyu_model_single  | 0.867 | 0.817 |
| end2end           | 0.483 | 0.483 |
| frcnn_cad         | 0.733 | 0.627 |

##### Breast level:

| Model | ROC AUC  | PR AUC |
| ----------------- | ----- | ----- |
| nyu_glam          | 0.733 | 0.461 |
| nyu_gmic          | 0.867 | 0.85  |
| nyu_model         | 0.867 | 0.85  |
| nyu_model_single  | 0.933 | 0.903 |
| end2end           | 0.6   | 0.372 |
| frcnn_cad         | 0.667 | 0.622 |

### I am getting `ValueError: unsupported pickle protocol: 5`
The reason for this error is when pickled data list file (e.g. data.pkl) is saved with Python 3.8 or later and highest (5) protocol. Models in the metarepository do not have the support of pickle protocol 5. Please save your data list file with protocol 4 or 3, e.g. `pickle.dump(datalist_dictionary, file, protocol=4)`.


## Submission Policy
## Reference

