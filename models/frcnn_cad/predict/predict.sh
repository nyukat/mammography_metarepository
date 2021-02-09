#!/bin/bash

# Generate predictions using the frcnn_cad model

CONTAINER_PICKLE_FILE=$1
CONTAINER_PATH_TO_IMAGES=$2
CONTAINER_PREDICTION_FILE=$3
DEVICE=$5

python3 generate_predictions.py --exam-list-path "${CONTAINER_PICKLE_FILE}" --input-data-folder "${CONTAINER_PATH_TO_IMAGES}" --prediction-file "${CONTAINER_PREDICTION_FILE}" --use-gpu "${DEVICE}"
