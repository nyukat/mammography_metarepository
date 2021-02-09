# Script for creating image-level predictions
# Iterate over all of the images - performing cropping, extracting centers, generating heatmaps, run classifiers

import pickle
import sys

import numpy as np
import pandas as pd
from src.cropping.crop_single import crop_single_mammogram
from src.heatmaps.run_producer_single import produce_heatmaps
from src.modeling.run_model_single import run
from src.optimal_centers.get_optimal_center_single import get_optimal_center_single
from tqdm import tqdm


def main(pkl_file, image_path, device_type, prediction_file, num_epochs, heatmap_batch_size, use_heatmaps):
    random_number_generator = np.random.RandomState(0)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    prediction_df = pd.DataFrame()
    image_indexes = []
    malignant_pred = []
    malignant_label = []
    for d in tqdm(data):
        for v in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            if d[v] is None or len(d[v]) == 0:
                continue
            else:
                if v[0] == 'L':
                    malignant_label.append(d['cancer_label']['left_malignant'])
                else:
                    malignant_label.append(d['cancer_label']['right_malignant'])

                index = random_number_generator.randint(low=0, high=len(d[v]))
                image_id = d[v][index]
                image_indexes.append(image_id)
                mammogram_path = image_path + '/' + image_id + '.png'
                cropped_path = "/home/bcc/breast_cancer_classifier/cropped.png"
                metadata_path = "/home/bcc/breast_cancer_classifier/cropped_metadata.pkl"
                # Crop mammogram
                crop_single_mammogram(mammogram_path=mammogram_path,
                                      view=v,
                                      horizontal_flip=d['horizontal_flip'],
                                      cropped_mammogram_path=cropped_path,
                                      metadata_path=metadata_path,
                                      num_iterations=100,
                                      buffer_size=50
                                      )
                # Optimal center
                get_optimal_center_single(
                    cropped_mammogram_path=cropped_path,
                    metadata_path=metadata_path
                )
                heatmap_path_malignant = "/home/bcc/breast_cancer_classifier/malignant_heatmap.hdf5"
                heatmap_path_benign = "/home/bcc/breast_cancer_classifier/benign_heatmap.hdf5"
                if use_heatmaps:
                    # Heatmap generation
                    heatmap_parameters = dict(
                        device_type=device_type,
                        gpu_number=0,
                        patch_size=256,
                        stride_fixed=70,
                        more_patches=5,
                        minibatch_size=heatmap_batch_size,
                        seed=0,
                        initial_parameters="models/sample_patch_model.p",
                        input_channels=3,
                        number_of_classes=4,
                        cropped_mammogram_path=cropped_path,
                        metadata_path=metadata_path,
                        heatmap_path_malignant=heatmap_path_malignant,
                        heatmap_path_benign=heatmap_path_benign,
                        heatmap_type=[0, 1],
                        use_hdf5=True
                    )
                    produce_heatmaps(heatmap_parameters)

                image_and_heatmap_parameters = dict(
                    view=v,
                    model_path="models/ImageHeatmaps__ModeImage_weights.p" if use_heatmaps else "models/ImageOnly__ModeImage_weights.p",
                    cropped_mammogram_path=cropped_path,
                    metadata_path=metadata_path,
                    device_type=device_type,
                    gpu_number=0,
                    max_crop_noise=(100, 100),
                    max_crop_size_noise=100,
                    batch_size=1,
                    seed=0,
                    augmentation=True,
                    num_epochs=num_epochs,
                    use_heatmaps=use_heatmaps,
                    heatmap_path_benign=heatmap_path_benign if use_heatmaps else None,
                    heatmap_path_malignant=heatmap_path_malignant if use_heatmaps else None,
                    use_hdf5=True
                )
                score_dict = run(image_and_heatmap_parameters)
                malignant_pred.append(score_dict['malignant'])

    prediction_df["image_index"] = image_indexes
    prediction_df["malignant_pred"] = malignant_pred
    prediction_df["malignant_label"] = malignant_label
    prediction_df.to_csv(prediction_file)


if __name__ == "__main__":
    pkl_file = sys.argv[1]
    image_path = sys.argv[2]
    device_type = sys.argv[3]
    prediction_file = sys.argv[4]
    num_epochs = int(sys.argv[5])
    heatmap_batch_size = int(sys.argv[6])
    use_heatmaps_str = sys.argv[7]
    if use_heatmaps_str == "True":
        use_heatmaps = True
    else:
        use_heatmaps = False
    main(pkl_file, image_path, device_type, prediction_file, num_epochs, heatmap_batch_size, use_heatmaps)
