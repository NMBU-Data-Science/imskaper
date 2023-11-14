# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:40:03 2021

@author: Nasibeh Mohammadi
@email: nasibeh.mohammadi@gmail.com
"""
import numpy as np
import pandas as pd
import six
from collections import Counter
import nibabel as nib
import pkg_resources


class LBPFeature():
    """
    This code calculates LBP number of each cell in a 3d image with regards
    to the given mask.
    For fast computation, first making shift vectors including six vectors
    such as :
        x = 1, y=0, z=0
        x = -1, y=0, z=0
        x = 0, y=1, z=0
        x = 0, y=-1, z=0
        x = 0, y=0, z=1
        x = 0, y=0, z=-1
    By shifting the image with these shift vectors we can compare each cell
    with its direct neighbours

    In this code P=6 and R=1, the neighbours on diagonal were not considered.

    After calculating the LBP number for each cell in the masked area of image,
    the LBP feature vector to be used in radiomics study was constructed.

    :param image_name: the image file name and its corresponding path
    :param mask_name: the mask file name and its corresponding path

    """

    def __init__(self, image_name, mask_name, label=1):
        if type(image_name) == str:
            # load image and mask.
            image_load = nib.load(image_name)
            mask_load = nib.load(mask_name)
            # get the array of image and mask .
            image = image_load.get_fdata()
            mask = mask_load.get_fdata()
        else:
            image = image_name
            mask = mask_name

       # self.image_name=image_name
       # self.mask_name=mask_name
        self.image = image
        self.mask = mask

        if '__iter__' in dir(label):
            for label_val in label:
                self.mask[mask == label_val] = 1
            label = 1

        self.label = label
        n, m, k = self.image.shape
        self.n = n
        self.m = m
        self.k = k
        # read the pattern  file which is based on rotation invariant concept.
        self.pattern = pd.read_csv(pkg_resources.resource_stream(__name__, 'rotation_invariant_pattern.txt'), sep="\t",
                                   converters={'rotation_invariant': lambda x: str(x)})

    def shift(self, dx, dy, dz):
        # extend the zone to all directions with zeros.
        extended_square = np.zeros(
            (3*self.n, 3*self.m, 3*self.k), self.image.dtype)
        extended_square[self.n:self.n+self.n, self.m:self.m+self.m,
                        self.k:self.k+self.k] = self.image
        x = self.n+dy
        y = self.m-dx
        z = self.k-dz
        return extended_square[x:x+self.n, y:y+self.m, z:z+self.k]

    def feature_vector(self):
        # we need permutation of three numbers -1, 0, 1 to make shift vectors.
        x = [-1, 1, 0]
        y = [-1, 1, 0]
        z = [-1, 1, 0]
        x = pd.DataFrame(data={'x': x}, index=np.repeat(0, len(x)))
        y = pd.DataFrame({'y': y}, index=np.repeat(0, len(y)))
        z = pd.DataFrame({'z': z}, index=np.repeat(0, len(z)))
        # get all permutations stored in a new df.
        shift_vec = pd.merge(x, (pd.merge(y, z, left_index=True,
                                          right_index=True)), left_index=True, right_index=True)

        # becuase we only need direct neighbors, we only want the shift vectors
        # that have only one value (either 1 or -1) and the two other values
        # should be zero.
        shift_vec = shift_vec[shift_vec.abs().sum(axis=1) == 1]

        # shift list are intensity values of cells when we shifted the image
        # based on the corresponding shift vector in the extended square that
        # we made earlier.
        shift_list = []
        for i in range(len(shift_vec)):
            dx = shift_vec.iloc[i]['x']
            dy = shift_vec.iloc[i]['y']
            dz = shift_vec.iloc[i]['z']
            shift_list.append(self.shift(dx, dy, dz))

        # Calculating LBP for each cell
        result = np.zeros((self.n, self.m, self.k), self.image.dtype)
        for x_center in range(self.n):
            for y_center in range(self.m):
                for z_center in range(self.k):
                    # only calculating LBP for the mask region has value 1 (self.label)
                    if self.mask[x_center, y_center, z_center] == self.label:
                        decimal_center = 0
                        for j in range(len(shift_list)):
                            # Comparing the image with corresponding shifted area.
                            # b=np.sign(
                            #     shift_list[j][x_center][y_center][z_center] -
                            #     self.image[x_center][y_center][z_center])
                            # # assign 0 or 1 to the corresponding neighbour.
                            # if b == -1:
                            #     sign=0
                            # else:
                            #     sign= 1

                            # MODIFICATION from Nasibeh's code
                            compare = shift_list[j][x_center][y_center][z_center] < self.image[x_center][y_center][z_center]
                            sign = 0 if compare else 1

                            decimal_center += sign*(2**j)

                        # mapping each decimal value to the corresponding
                        # pattern based on rotation invariant concept.
                        if np.where((
                                self.pattern['original'] == decimal_center
                        ).values == True):

                            result[x_center][y_center][z_center] = self.pattern.loc[
                                self.pattern['original'] == decimal_center]['minvalue'].values[0]

        # removing the zero regions (non-label).
        mask_result = result[self.mask == self.label]

        # calculating each pattern frequency.
        frequency = Counter(mask_result)

        # make the feature vector
        freq_total = sum(frequency.values())
        lbp_feature_vect_dict = {i: 0 for i in np.unique(
            self.pattern['rotation_invariant'].values)}
        for (label, val) in six.iteritems(frequency):
            if np.where((self.pattern['original'] == label).values == True):
                key = self.pattern.loc[self.pattern['original'] ==
                                       label]['rotation_invariant'].values[0]
                freq_scaled = val/freq_total
                lbp_feature_vect_dict[key] = freq_scaled

        return lbp_feature_vect_dict
