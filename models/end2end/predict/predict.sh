#!/bin/bash

# Generate predictions using the end-to-end fully convolutional model

CONTAINER_PICKLE_FILE=$1
CONTAINER_PATH_TO_IMAGES=$2
CONTAINER_PREDICTION_FILE=$3
NAME=$4
DEVICE=$5
PREPROCESSED_FOLDER="$6/end2end_${NAME}_preprocessed_images"

MEAN_PIXEL_INTENSITY="44.4"
MODEL="inbreast_vgg16_[512-512-1024]x2_hybrid.h5"
NUM_PROCESSES="10"
PREPROCESS_FLAG="True"

if [ "${PREPROCESS_FLAG}" = "True" ]; then
    [ -d "${PREPROCESSED_FOLDER}" ] && rm -r "${PREPROCESSED_FOLDER}"
fi

# Choose correct virtualenv
if [ "${DEVICE}" == "gpu" ]; then
    echo "Activating GPU venv."
    source /opt/venv-gpu/bin/activate
elif [ "${DEVICE}" == "cpu" ]; then
    echo "Activating CPU venv."
    source /opt/venv-cpu/bin/activate
fi

# Preprocess the images
python3 preprocess.py --exam-list-path "${CONTAINER_PICKLE_FILE}" --input-data-folder "${CONTAINER_PATH_TO_IMAGES}" --preprocessed-folder "${PREPROCESSED_FOLDER}" --num-processes "${NUM_PROCESSES}" --always-preprocess "${PREPROCESS_FLAG}" --image-format png

# Generate predictions
python3 generate_predictions.py --exam-list-path "${CONTAINER_PICKLE_FILE}" --input-data-folder "${PREPROCESSED_FOLDER}" --prediction-file "${CONTAINER_PREDICTION_FILE}" --model "${MODEL}" --rescale-factor 0.003891 --mean-pixel-intensity "${MEAN_PIXEL_INTENSITY}"
