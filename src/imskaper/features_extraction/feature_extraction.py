# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features extraction script.
"""

__author__ = "Ahmed Albuni, Ngoc Huynh"
__email__ = "ahmed.albuni@gmail.com, ngoc.huynh.bao@nmbu.no"


import argparse
import logging
from csv import DictWriter
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from radiomics import (
    firstorder,
    glcm,
    gldm,
    glrlm,
    glszm,
    ngtdm,
    shape,
    shape2D,
)
from tqdm import tqdm
import click
from .LBP3d import LBPFeature

#  List of features groups available in pyradiomics package
#  This list match the input csv parameters file
FEATURES_LIST = (
    "shape",
    "first_order",
    "glszm",
    "glrlm",
    "ngtdm",
    "gldm",
    "glcm",
    "LBP"
)


def extract_radiomics_features(
    features_list,
    bin_count,
    images_path,
    masks_path=None,
    glcm_distance=None,
    ngtdm_distance=None,
    gldm_distance=None,
    gldm_a=0,
    output_file_name="output",
    label=1,
    bin_setting_name='binCount'
):
    """
    :param features_list: list of features to be extracted
    :param bin_count:
    :param images_path: The path that contains the images
    :param masks_path: The path of the masks, masks name should match the
    images names
    :param glcm_distance: A list of distances for GLCM calculations,
    default is [1]
    :param ngtdm_distance: List of integers. This specifies the distances
     between the center voxel and the neighbor, for which angles should be
      generated.
    :param gldm_distance: List of integers. This specifies the distances
     between the center voxel and the neighbor, for which angles should be
      generated.
    :param gldm_a:  integer, α cutoff value for dependence.
    A neighbouring voxel with gray level j is considered
    dependent on center voxel with gray level i if |i−j|≤α
    :param output_file_name: Name of the output csv file
    :return:
    """
    if glcm_distance is None:
        glcm_distance = [1]
    if ngtdm_distance is None:
        ngtdm_distance = [1]
    if gldm_distance is None:
        gldm_distance = [1]

    list_of_images = [
        f for f in listdir(images_path) if isfile(join(images_path, f))
    ]

    bin_settings = {
        bin_setting_name: bin_count
    }

    for i, img in tqdm(
        enumerate(list_of_images), total=len(list_of_images), unit="files"
    ):
        image_name = images_path + img
        image = sitk.ReadImage(image_name)

        row = dict()
        row["Name"] = img

        #  If the mask is not available we create a dummy mask here that
        #  covers the whole image
        if masks_path is None:
            mask = np.zeros((sitk.GetArrayFromImage(image)).shape, int) + 1
            mask = sitk.GetImageFromArray(mask)

        else:
            mask_name = masks_path + img
            mask = sitk.ReadImage(mask_name)
            if type(label) == list:
                # merge all labels
                # label = 1
                labels = label.copy()
                label = 1

                mask_data = sitk.GetArrayFromImage(mask)
                for label_val in labels:
                    mask_data[mask_data==label_val] = 1
                # regenerate the sitk image obj (for voxel size, etc..)
                new_mask = sitk.GetImageFromArray(mask_data)
                new_mask.CopyInformation(mask)
                mask = new_mask


            # Shape features applied only when the mask is provided
            if "shape" in features_list:
                if len((sitk.GetArrayFromImage(image)).shape) == 2:
                    shape_2d_f = shape2D.RadiomicsShape2D(
                        image, mask, label=label, **bin_settings
                    )
                    row.update(get_selected_features(shape_2d_f, "shape_2d"))
                else:
                    shape_f = shape.RadiomicsShape(
                        image, mask, label=label, **bin_settings
                    )
                    row.update(get_selected_features(shape_f, "shape"))

        if "first_order" in features_list:
            f_o_f = firstorder.RadiomicsFirstOrder(
                image, mask, label=label, **bin_settings
            )
            row.update(get_selected_features(f_o_f, "first_order"))
        if "glszm" in features_list:
            glszm_f = glszm.RadiomicsGLSZM(image, mask, label=label, **bin_settings)
            row.update(get_selected_features(glszm_f, "glszm"))
        if "glrlm" in features_list:
            glrlm_f = glrlm.RadiomicsGLRLM(image, mask, label=label, **bin_settings)
            row.update(get_selected_features(glrlm_f, "glrlm"))
        if "ngtdm" in features_list:
            for d in ngtdm_distance:
                ngtdm_f = ngtdm.RadiomicsNGTDM(
                    image, mask, distances=[d], label=label, **bin_settings
                )
                row.update(
                    get_selected_features(
                        ngtdm_f, "ngtdm", additional_param="_d_" + str(d)
                    )
                )
        if "gldm" in features_list:
            for d in gldm_distance:
                gldm_f = gldm.RadiomicsGLDM(
                    image,
                    mask,
                    distances=[d],
                    gldm_a=gldm_a,
                    label=label,
                    **bin_settings
                )
                row.update(
                    get_selected_features(
                        gldm_f, "gldm", additional_param="_d_" + str(d)
                    )
                )
        if "glcm" in features_list:
            for d in glcm_distance:
                glcm_f = glcm.RadiomicsGLCM(
                    image, mask, distances=[d], label=label, **bin_settings
                )
                row.update(
                    get_selected_features(
                        glcm_f, "glcm", additional_param="_d_" + str(d)
                    )
                )
        if "LBP" in features_list:
            lbp_f = LBPFeature(image_name=sitk.GetArrayFromImage(image), mask_name=sitk.GetArrayFromImage(mask), label=label).feature_vector()
            row.update(
                get_selected_features(
                    lbp_f,"LBP"
                )
            )

        if i == 0:
            create_output_file(output_file_name + ".csv", row.keys())
        add_row_of_data(output_file_name + ".csv", row.keys(), row)


def create_output_file(file_name, columns):
    with open(file_name, "w", newline="") as f:
        writer = DictWriter(f, fieldnames=columns)
        writer.writeheader()


def add_row_of_data(file_name, columns, row):
    with open(file_name, "a", newline="") as f:
        writer = DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def get_selected_features(selected_feature, category, additional_param=None):
    data = {}
    if category == 'LBP':
        for (key, val) in six.iteritems(selected_feature):
            key = category + "_" + str(key)
            data[key] = val
    else:
        selected_feature.execute()
        for (key, val) in six.iteritems(selected_feature.featureValues):
            key = category + "_" + key
            if additional_param is not None:
                key = key + additional_param
            data[key] = val

    return data


def imskaper_feature_extract():
    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Features extraction")
    parser.add_argument(
        "-file", type=str, help="CSV parameters file name and " "path"
    )
    parser.add_argument(
        "-glcm_distance",
        type=str,
        help="list of distances, " "comma separated. " "default: 1",
    )
    parser.add_argument(
        "-ngtdm_distance",
        type=str,
        help="list of distances, " "comma separated. " "default 1",
    )
    parser.add_argument(
        "-gldm_distance",
        type=str,
        help="list of distances, " "comma separated. " "default 1",
    )
    parser.add_argument(
        "-gldm_a", type=int, help="Cutoff value for dependence, " "default: 0"
    )

    args = parser.parse_args()
    glcm_d = args.glcm_distance
    if glcm_d is not None:
        glcm_d = glcm_d.split(",")
    ngtdm_d = args.ngtdm_distance
    if ngtdm_d is not None:
        ngtdm_d = ngtdm_d.split(",")
    gldm_d = args.gldm_distance
    if gldm_d is not None:
        gldm_d = gldm_d.split(",")

    gldm_a = args.gldm_a
    if gldm_a is None:
        gldm_a = 0

    if not args.file:
        print('A path to the template file must be specified using the -file argument.')
        print('imskaper_feature_extraction -file <path_to_csv_template>')
        return

    f_list = pd.read_csv(args.file)

    for index, row in f_list.iterrows():
        print("Output file: ", row["output_file_name"])
        feature = []
        for f in FEATURES_LIST:
            if row[f] == 1:
                feature.append(f)
        if type(row["mask_dir"]) is not str:
            mask_path = None
        else:
            mask_path = row["mask_dir"]

        label = row.get('label', 1)
        if type(label) is not int:
            try:
                label = int(label)
            except:
                label = list(map(int, label.split(',')))

        if row.get('bin_count') is None or np.isnan(row.get('bin_count')):
            bin_setting = 'bin_width'
            bin_setting_name = 'binWidth'
        else:
            bin_setting = 'bin_count'
            bin_setting_name = 'binCount'

        extract_radiomics_features(
            feature,
            row[bin_setting],
            row["image_dir"],
            mask_path,
            output_file_name=row["output_file_name"],
            glcm_distance=glcm_d,
            ngtdm_distance=ngtdm_d,
            gldm_distance=gldm_d,
            gldm_a=gldm_a,
            label=label,
            bin_setting_name=bin_setting_name
        )

if __name__ == "__main__":
    imskaper_feature_extract()
