#!/bin/bash

CONTAINER_PICKLE_FILE=$1
CONTAINER_PATH_TO_IMAGES=$2
CONTAINER_PREDICTION_FILE=$3
NAME=$4
DEVICE=$5
PREPROCESSED_FOLDER=$6

MODEL_NAME="model_joint"
NUM_PROCESSES="10"
PREPROCESS_FLAG="True"

MODEL_PATH="models/"
CROPPED_IMAGE_PATH="${PREPROCESSED_FOLDER}/nyu_glam_${NAME}_cropped_images"
CROPPED_EXAM_LIST_PATH="${PREPROCESSED_FOLDER}/nyu_glam_${NAME}_cropped_images/cropped_exam_list.pkl"
OUTPUT_PATH="${PREPROCESSED_FOLDER}/nyu_glam_${NAME}_sample_output"
EXAM_LIST_PATH="${PREPROCESSED_FOLDER}/nyu_glam_${NAME}_center_data.pkl"

if [ "${PREPROCESS_FLAG}" = "True" ]; then
    [ -d "${CROPPED_IMAGE_PATH}" ] && rm -r "${CROPPED_IMAGE_PATH}"
    [ -d "${OUTPUT_PATH}" ] && rm -r "${OUTPUT_PATH}"
    [ -d "${EXAM_LIST_PATH}" ] && rm -r "${EXAM_LIST_PATH}"
fi

export PYTHONPATH=$(pwd):$PYTHONPATH

echo -e "\nStage 1: Cropping Mammograms.."
python3 src/cropping/crop_mammogram.py \
    --input-data-folder "${CONTAINER_PATH_TO_IMAGES}" \
    --output-data-folder "${CROPPED_IMAGE_PATH}" \
    --exam-list-path "${CONTAINER_PICKLE_FILE}" \
    --cropped-exam-list-path "${CROPPED_EXAM_LIST_PATH}" \
    --num-processes $NUM_PROCESSES

echo -e "\nStage 2: Extracting Centers.."
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path "${CROPPED_EXAM_LIST_PATH}" \
    --data-prefix "${CROPPED_IMAGE_PATH}" \
    --output-exam-list-path "${EXAM_LIST_PATH}" \
    --num-processes $NUM_PROCESSES

echo -e "\nStage 3: Running Classifier.."
python3 src/scripts/run_model.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${EXAM_LIST_PATH}" \
    --image-path "${CROPPED_IMAGE_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --segmentation-path "None" \
    --device-type "${DEVICE}" \
    --gpu-number 0 \
    --model-index "${MODEL_NAME}"

echo "Copying results into the output csv file."
cp "${OUTPUT_PATH}/predictions.csv" "${CONTAINER_PREDICTION_FILE}"
