# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Defines utility functions for managing the dataset.
"""
from src.constants import VIEWS


def unpack_exam_into_images(exam_list, cropped=False):
    """
    Turns exam_list into image_list for parallel functions which process each image separately.
    """
    image_list = []
    for i, exam in enumerate(exam_list):
        for view in VIEWS.LIST:
            for j, image in enumerate(exam[view]):
                image_dict = dict(
                    short_file_path=image,
                    horizontal_flip=exam['horizontal_flip'],
                    full_view=view,
                    side=view[0],
                    view=view[2:],
                )
                if cropped:
                    image_dict["window_location"] = exam['window_location'][view][j]
                    image_dict["rightmost_points"] = exam['rightmost_points'][view][j]
                    image_dict["bottommost_points"] = exam['bottommost_points'][view][j]
                    image_dict["distance_from_starting_side"] = exam['distance_from_starting_side'][view][j]
                image_list.append(image_dict)
    return image_list


def add_metadata(exam_list, additional_metadata_name, additional_metadata_dict):
    """
    Appends new information about images into exam_list.
    """
    for exam in exam_list:
        assert additional_metadata_name not in exam, f"This metadata ({additional_metadata_name}) is already included."
        exam[additional_metadata_name] = dict()
        for view in VIEWS.LIST:
            exam[additional_metadata_name][view] = []
            for j, image in enumerate(exam[view]):
                exam[additional_metadata_name][view].append(additional_metadata_dict[image])
