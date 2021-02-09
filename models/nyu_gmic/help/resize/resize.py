import argparse
import multiprocessing as mp
import os
import time
from functools import partial

import imageio
import numpy as np
import src.data_loading.loading as loading
import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
from src.constants import VIEWS


def flip_image(image, view, horizontal_flip):
    """
    If training mode, makes all iamges face right direction.
    In medical, keeps the original directions unless horizontal_flip is set.
    """
    if horizontal_flip == 'NO':
        if VIEWS.is_right(view):
            image = np.fliplr(image)
    elif horizontal_flip == 'YES':
        if VIEWS.is_left(view):
            image = np.fliplr(image)
    return image


def resize(input_data_folder, exam_list_path, output_data_folder, num_processes):
    # Random number generator to satsify function call
    rng = np.random.RandomState(0)

    # Make output directory
    os.makedirs(output_data_folder, exist_ok=True)

    # Load in exam_list
    exam_list = pickling.unpickle_from_file(exam_list_path)

    # Create partial function 
    resize_one_exam_func = partial(resize_one_exam, input_data_folder=input_data_folder, output_data_folder=output_data_folder, rng=rng)

    # Create multiprocessing
    start = time.time()
    with mp.Pool(num_processes) as p:
        p.map(resize_one_exam_func, exam_list)
    end = time.time()


# Load in image based on path
def resize_one_exam(exam, input_data_folder, output_data_folder, rng):
    for view in VIEWS.LIST:
        for short_file_path in exam[view]:
            loaded_image = reading_images.read_image_png(os.path.join(input_data_folder, short_file_path + '.png'))
            # Flip image
            loaded_image = flip_image(loaded_image, view, exam["horizontal_flip"])
            # Do the resizing
            cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                image=loaded_image,
                auxiliary_image=None,
                view=view,
                random_number_generator=rng,
                best_center=exam["best_center"][view][0],
                augmentation=False,
                max_crop_noise=(0, 0),
                max_crop_size_noise=0
            )
            # Save image
            save_path = os.path.join(output_data_folder, short_file_path + '.png')
            folders = save_path.split('/')
            dirs = folders[:-1]
            dir_path = '/'.join(dirs)
            os.makedirs(dir_path, exist_ok=True)
            imageio.imwrite(save_path, cropped_image)
    # Call random_augmentation_best_center
    # Get resize image
    # Save it


# Load cropped images
# Resize images
# Save images
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images for model")
    parser.add_argument('--input-data-folder', required=True)
    parser.add_argument('--output-data-folder', required=True)
    parser.add_argument('--cropped-exam-list-path', required=True)
    parser.add_argument('--num-processes', default=10, type=int)
    args = parser.parse_args()

    # Partial function binding input-data-folder, output-data-folder, cropped-exam-list-path
    # Then use with Pool
    # Or rewrite resize
    resize(
        input_data_folder=args.input_data_folder,
        exam_list_path=args.cropped_exam_list_path,
        output_data_folder=args.output_data_folder,
        num_processes=args.num_processes,
    )
