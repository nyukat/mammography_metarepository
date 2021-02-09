#!/bin/bash

# Generate predictions using the breast level NYU Breast Cancer Classifier model

CONTAINER_PICKLE_FILE=$1
CONTAINER_PATH_TO_IMAGES=$2
CONTAINER_PREDICTION_FILE=$3
NAME=$4
DEVICE=$5
PREPROCESSED_FOLDER=$6

HEATMAP_BATCH_SIZE="100"
NUM_EPOCHS="10"
NUM_PROCESSES="10"
PREPROCESS_FLAG="True"
USE_HEATMAPS="False"

PATCH_MODEL_PATH="models/sample_patch_model.p"
IMAGE_MODEL_PATH="models/sample_image_model.p"
IMAGE_HEATMAPS_MODEL_PATH="models/sample_imageheatmaps_model.p"

CROPPED_IMAGE_PATH="${PREPROCESSED_FOLDER}/nyu_model_${NAME}_cropped_images"
CROPPED_EXAM_LIST_PATH="${PREPROCESSED_FOLDER}/nyu_model_${NAME}_cropped_images/cropped_exam_list.pkl"
EXAM_LIST_PATH="${PREPROCESSED_FOLDER}/nyu_model_${NAME}_center_data.pkl"
HEATMAPS_PATH="${PREPROCESSED_FOLDER}/nyu_model${NAME}_heatmaps"
echo "Heatmaps path: ${HEATMAPS_PATH}"

if [ "${PREPROCESS_FLAG}" = "True" ]; then
    [ -d "${CROPPED_IMAGE_PATH}" ] && rm -r "${CROPPED_IMAGE_PATH}"
    [ -d "${HEATMAPS_PATH}" ] && rm -r "${HEATMAPS_PATH}"
fi

PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTHONPATH

echo -e "\nStage 1: Crop Mammograms"
if [ "${PREPROCESS_FLAG}" = "False" ] && [ -d "${CROPPED_IMAGE_PATH}" ]
then
    echo "The images have already been cropped. Skipping cropping."
else
    echo "Data Folder: ${CONTAINER_PATH_TO_IMAGES}"
    echo "Initial Exam List Path: ${CONTAINER_PICKLE_FILE}"
    python3 src/cropping/crop_mammogram.py \
        --input-data-folder "${CONTAINER_PATH_TO_IMAGES}" \
        --output-data-folder "${CROPPED_IMAGE_PATH}" \
        --exam-list-path "${CONTAINER_PICKLE_FILE}" \
        --cropped-exam-list-path "${CROPPED_EXAM_LIST_PATH}"  \
        --num-processes "${NUM_PROCESSES}"
fi

echo -e "\nStage 2: Extract Centers"
if [ "${PREPROCESS_FLAG}" = "False" ] && [ -f "${EXAM_LIST_PATH}" ]
then
    echo "The image centers have already been extracted. Skipping centers."
else
    python3 src/optimal_centers/get_optimal_centers.py \
        --cropped-exam-list-path "${CROPPED_EXAM_LIST_PATH}" \
        --data-prefix "${CROPPED_IMAGE_PATH}" \
        --output-exam-list-path "${EXAM_LIST_PATH}" \
        --num-processes "${NUM_PROCESSES}"
fi

echo -e "\nStage 3: Generate Heatmaps"
if [ "${USE_HEATMAPS}" != "True" ]
then
    echo "Heatmap generation set to false. Skipping."
elif [ "${USE_PRIOR_PREPROCESSING_IF_EXISTS}" = "True" ] && [ -d "${HEATMAPS_PATH}" ]
then
    echo "Skipping heatmaps. They are already created."
else
    python3 src/heatmaps/run_producer.py \
        --model-path "${PATCH_MODEL_PATH}" \
        --data-path "${EXAM_LIST_PATH}" \
        --image-path "${CROPPED_IMAGE_PATH}" \
        --batch-size "${HEATMAP_BATCH_SIZE}" \
        --output-heatmap-path "${HEATMAPS_PATH}" \
        --device-type "${DEVICE}" \
        --gpu-number 0
fi

if [ "${USE_HEATMAPS}" != "True" ]
then
    echo -e "\nStage 4a: Run Classifier (Image)"
    python3 src/modeling/run_model.py \
        --model-path "${IMAGE_MODEL_PATH}" \
        --data-path "${EXAM_LIST_PATH}" \
        --image-path "${CROPPED_IMAGE_PATH}" \
        --output-path "${CONTAINER_PREDICTION_FILE}" \
        --use-augmentation \
        --num-epochs "${NUM_EPOCHS}" \
        --device-type "${DEVICE}" \
        --gpu-number 0
fi

if [ "${USE_HEATMAPS}" = "True" ]
then
    echo -e "\nStage 4b: Run Classifier (Image+Heatmaps)"
    python3 src/modeling/run_model.py \
        --model-path "${IMAGE_HEATMAPS_MODEL_PATH}" \
        --data-path "${EXAM_LIST_PATH}" \
        --image-path "${CROPPED_IMAGE_PATH}" \
        --output-path "${CONTAINER_PREDICTION_FILE}" \
        --use-heatmaps \
        --heatmaps-path "${HEATMAPS_PATH}" \
        --use-augmentation \
        --num-epochs "${NUM_EPOCHS}" \
        --device-type "${DEVICE}" \
        --gpu-number 0
fi
