#!/bin/bash

MODEL=$1  # Name of the model, currently supported: `end2end`, `frcnn_cad`, `nyu_gmic`, `nyu_model`, `nyu_model_single`
NAME=$2  # Name for experiment - used to store preprocessed images
LOCAL_PATH_TO_IMAGES=$3  # Attach as volume - where the images are stored
LOCAL_PATH_TO_PREPROCESSED_IMAGES=$4 # Attach as volume - where the preprocessed images will be stored. They will be stored under the NAME_EXPERIMENT pattern
LOCAL_PICKLE_FILE=$5  # Copy into container before generating predictions - where the pickle file is located
LOCAL_PREDICTION_PATH=$6  # Copy container prediction file to local prediction file
DEVICE=$7  # gpu or cpu
GPU=$8  # GPU to execute on
BOOTSTRAPPING=$9  # whether to use bootstrapping

get_groupname () {
    ENTRIES=()

    while read line; do
        if [[ ! -z "$line" ]]
        then
            ENTRIES+=("$line")
        fi
    done < users.txt

    GROUP_INFO=($(echo ${ENTRIES[0]} | tr "," " "))
    GROUPNAME=${GROUP_INFO[0]}
    GROUPID=${GROUP_INFO[1]}

    echo ${GROUPNAME}
}

expand_if_absolute () {
  # Check if path exists, and expand to absolute path
  # First argument - path to potentially expand
  # Second argument - whether the path must exist (if true, passing path that does not exist will result in an error)
  if ! [[ -f "${1}" ||  -d "${1}" ]]; then
      if [ "${2}" = "true" ]; then
          echo "${1} does not exist, please check. Exiting."
          exit
      else
          mkdir -p "${1}"
      fi
  fi
  echo "$(cd "$(dirname "${1}")" || exit ; pwd)/$(basename "${1}")"
}

echo "Checking if users.txt exists"
# Check if users.txt exists. It should be created from template_users.txt.
if [[ ! -f users.txt ]]; then
    echo "users.txt does not exist, please check README.md on how to create it. Exiting."
    exit
fi

echo "Expand preprocessed image path"
ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES=$(expand_if_absolute "${LOCAL_PATH_TO_PREPROCESSED_IMAGES}" "false")
if [ "${ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES: -8}" = "Exiting." ]; then
    echo "${ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES}"
    exit
fi

echo "Creating subdirectory for experiment"
# Check if subdirectory exists in preprocessing path. If not create one.
if [[ ! -d "${ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES}/${MODEL}_${NAME}_preprocessing" ]]; then
    mkdir "${ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES}/${MODEL}_${NAME}_preprocessing"
else
    echo "Directory already exists"
fi

echo "Expanding path to images"
ABSOLUTE_PATH_TO_IMAGES=$(expand_if_absolute "${LOCAL_PATH_TO_IMAGES}" "true")
if [ "${ABSOLUTE_PATH_TO_IMAGES: -8}" = "Exiting." ]; then
    echo "${ABSOLUTE_PATH_TO_IMAGES}"
    exit
fi
ABSOLUTE_PICKLE_FILE=$(expand_if_absolute "${LOCAL_PICKLE_FILE}" "true")
if [ "${ABSOLUTE_PICKLE_FILE: -8}" = "Exiting." ]; then
    echo "${ABSOLUTE_PICKLE_FILE}"
    exit
fi
ABSOLUTE_PREDICTION_PATH=$(expand_if_absolute "${LOCAL_PREDICTION_PATH}" "false")

if ls models/"${MODEL}" &>/dev/null; then
    echo -e "\nUsed model: ${MODEL}."
else
    echo -e "\nModel ${MODEL} is not supported. Please use one of below models: \n$(ls models). Exitting."
    exit
fi

# Adding support for uppercase device names, to avoid popular errors
if [ "${DEVICE}" = "GPU" ]; then
    DEVICE="gpu"
fi
if [ "${DEVICE}" = "CPU" ]; then
    DEVICE="cpu"
fi

if [ "${DEVICE}" = "gpu" ]; then
    echo "Device: gpu, number: ${GPU}."
elif [ "${DEVICE}" = "cpu" ]; then
    if [ "${MODEL}" = "frcnn_cad" ]; then
        echo "frcnn_cad is not supported on CPU. Exiting."
        exit
    fi
    echo "Device: cpu."
else
    echo "Device type ${DEVICE} is incompatible. Please use either 'cpu' or 'gpu'. Exiting."
    exit
fi

echo "Name of the experiment: ${NAME}"
echo "Path to images: ${ABSOLUTE_PATH_TO_IMAGES}"
echo "Path to the pickle file: ${ABSOLUTE_PICKLE_FILE}"
echo -e "Path for saving predictions: ${ABSOLUTE_PREDICTION_PATH}\n"

LOCAL_PREDICTION_FILE="${ABSOLUTE_PREDICTION_PATH}/${MODEL}_${NAME}_predictions.csv"
CONTAINER_PATH_TO_IMAGES="/home/images"  # don't change this
CONTAINER_PATH_TO_PREPROCESSED_IMAGES="/home/preprocessed_images"
CONTAINER_PICKLE_FILE="/home/data.pkl"  # don't change this
CONTAINER_PREDICTION_FILE="/home/predictions/${MODEL}_${NAME}_predictions.csv"  # don't change this
# Extract where to put predict folder contents on container from config file for model
CONTAINER_PREDICT_PATH=$(sed '2q;d' "models/${MODEL}/config.txt" | cut -c 24- | tr -d '"')
# Extract extra arguments from config file
EXTRA_ARGS=($(tail -n +5 "./models/${MODEL}/config.txt" | tr '\n' ' '))


build_docker_image () {
    GROUPNAME=$(get_groupname)
    echo "${GROUPNAME}"
    echo "Building image"
    docker build -t "${MODEL}" -f models/${MODEL}/Dockerfile --build-arg GROUPNAME="${GROUPNAME}" .
}

run_model () {
    cd "./models/${MODEL}" || exit
    # modify parameters in predict.sh by expanding EXTRA_ARGS array
    echo "Model extra arguments:"
    for arg in "${EXTRA_ARGS[@]}"
    do
        ARG=$(echo "${arg}" | cut -d'=' -f 1)=
        VAL=$(echo "${arg}" | cut -d'=' -f 2)
        echo " - ${ARG}${VAL}"
        sed -i "s/$ARG.*/$ARG\"$VAL\"/1" ./predict/predict.sh
    done
    echo -e "\nStarting docker container for ${MODEL} model."
    # Start docker container for model on CPU or GPU
    if [ "${DEVICE}" = "gpu" ]; then
        docker container run -i -d -u $(whoami) -v "${ABSOLUTE_PATH_TO_IMAGES}:${CONTAINER_PATH_TO_IMAGES}:ro" -v "${ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES}/${MODEL}_${NAME}_preprocessing:${CONTAINER_PATH_TO_PREPROCESSED_IMAGES}" --network none --gpus device="${GPU}" "${MODEL}" bash
    elif [ "${DEVICE}" = "cpu" ]; then
        docker container run -i -d -u $(whoami) -v "${ABSOLUTE_PATH_TO_IMAGES}:${CONTAINER_PATH_TO_IMAGES}:ro" -v "${ABSOLUTE_PATH_TO_PREPROCESSED_IMAGES}/${MODEL}_${NAME}_preprocessing:${CONTAINER_PATH_TO_PREPROCESSED_IMAGES}" --network none "${MODEL}" bash
    fi
    # Store container id
    DOCKER_ALIAS=$(docker ps -l -q)
    docker container cp ./predict/. "${DOCKER_ALIAS}:${CONTAINER_PREDICT_PATH}"
    # Copy pickle file into container
    docker container cp "${ABSOLUTE_PICKLE_FILE}" "${DOCKER_ALIAS}:${CONTAINER_PICKLE_FILE}"
    # Resume running model - for debugging purposes
    #docker container exec -it "${DOCKER_ALIAS}" bash
    echo -e "\nRunning predict.sh for ${MODEL} model."
    docker container exec -it "${DOCKER_ALIAS}"  bash predict.sh "${CONTAINER_PICKLE_FILE}" "${CONTAINER_PATH_TO_IMAGES}" "${CONTAINER_PREDICTION_FILE}" "${NAME}" "${DEVICE}" "${CONTAINER_PATH_TO_PREPROCESSED_IMAGES}"
    # Copy predictions from container to local machine
    docker container cp "${DOCKER_ALIAS}:${CONTAINER_PREDICTION_FILE}" "${LOCAL_PREDICTION_FILE}"
    docker container stop "${DOCKER_ALIAS}"
    docker container rm "${DOCKER_ALIAS}"
    echo -e "\nEvaluating."
    cd ../..
    python3 ./evaluation/score.py "${ABSOLUTE_PICKLE_FILE}" "${LOCAL_PREDICTION_FILE}" "${BOOTSTRAPPING}"
    echo -e "\n"
}

if docker images | grep -q -w "${MODEL}"; then
    echo "Model already built, running."
    run_model
elif ls models/"${MODEL}" &>/dev/null; then
    echo "Model not built, building."
    build_docker_image
    run_model
fi
