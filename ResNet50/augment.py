# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise Exception('Python3 required')

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import skimage.io
import skimage.transform


def augment_image(img, mask=None, rotation_flag=False, reflection_flag=False,
                  jitter_augmentation_severity=0,  # jitter augmentation severity as a fraction of the image size
                  noise_augmentation_severity=0,  # noise augmentation as a percentage of current noise
                  scale_augmentation_severity=0,  # scale augmentation as a percentage of the image size):
                  blur_augmentation_max_sigma=0,  # blur augmentation kernel maximum size):
                  intensity_augmentation_severity=0):  # intensity augmentation as a percentage of the current intensity

    img = np.asarray(img)

    # ensure input images are np arrays
    img = np.asarray(img, dtype=np.float32)

    debug_worst_possible_transformation = False # useful for debuging how bad images can get

    # check that the input image and mask are 2D images
    assert len(img.shape) == 2 or len(img.shape) == 3

    # convert input Nones to expected
    if jitter_augmentation_severity is None:
        jitter_augmentation_severity = 0
    if noise_augmentation_severity is None:
        noise_augmentation_severity = 0
    if scale_augmentation_severity is None:
        scale_augmentation_severity = 0
    if blur_augmentation_max_sigma is None:
        blur_augmentation_max_sigma = 0
    if intensity_augmentation_severity is None:
        intensity_augmentation_severity = 0

    # confirm that severity is a float between [0,1]
    assert 0 <= jitter_augmentation_severity < 1
    assert 0 <= noise_augmentation_severity < 1
    assert 0 <= scale_augmentation_severity < 1
    assert 0 <= intensity_augmentation_severity < 1

    # get the size of the input image
    h, w, c = img.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        assert len(mask.shape) == 2 or len(mask.shape) == 3
        assert (mask.shape[0] == h and mask.shape[1] == w)

    # set default augmentation parameter values (which correspond to no transformation)
    orientation = 0
    reflect_x = False
    reflect_y = False
    jitter_x = 0
    jitter_y = 0
    scale_x = 1
    scale_y = 1

    if rotation_flag:
        orientation = 360 * np.random.rand()
    if reflection_flag:
        reflect_x = np.random.rand() > 0.5  # Bernoulli
        reflect_y = np.random.rand() > 0.5  # Bernoulli
    if jitter_augmentation_severity > 0:
        if debug_worst_possible_transformation:
            jitter_x = int(jitter_augmentation_severity * (w * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y
        else:
            jitter_x = int(jitter_augmentation_severity * (w * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y

    if scale_augmentation_severity > 0:
        max_val = 1 + scale_augmentation_severity
        min_val = 1 - scale_augmentation_severity
        if debug_worst_possible_transformation:
            scale_x = min_val + (max_val - min_val) * 1
            scale_y = min_val + (max_val - min_val) * 1
        else:
            scale_x = min_val + (max_val - min_val) * np.random.rand()
            scale_y = min_val + (max_val - min_val) * np.random.rand()

    # apply the affine transformation
    img = apply_affine_transformation(img, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)
    if mask is not None:
        mask = apply_affine_transformation(mask, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

    # apply augmentations
    if noise_augmentation_severity > 0:
        sigma_max = noise_augmentation_severity * (np.max(img) - np.min(img))
        max_val = sigma_max
        min_val = -sigma_max
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        sigma_img = np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * sigma
        img = img + sigma_img

    # apply blur augmentation
    if blur_augmentation_max_sigma > 0:
        max_val = blur_augmentation_max_sigma
        min_val = -blur_augmentation_max_sigma
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        if sigma < 0:
            sigma = 0
        if sigma > 0:
            img = scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect')

    if intensity_augmentation_severity > 0:
        img_range = np.max(img) - np.min(img)
        if debug_worst_possible_transformation:
            value = 1 * intensity_augmentation_severity * img_range
        else:
            value = np.random.rand() * intensity_augmentation_severity * img_range
        if np.random.rand() > 0.5:
            sign = 1.0
        else:
            sign = -1.0
        delta = sign * value
        img = img + delta # additive intensity adjustment

    img = np.asarray(img, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        mask = np.round(mask)
        return img, mask
    else:
        return img


def apply_affine_transformation(I, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y):

    if orientation is not 0:
        I = skimage.transform.rotate(I, orientation, preserve_range=True, mode='reflect')

    tform = skimage.transform.AffineTransform(translation=(jitter_x, jitter_y),
                                              scale=(scale_x, scale_y))
    I = skimage.transform.warp(I, tform._inv_matrix, mode='reflect', preserve_range=True)

    if reflect_x:
        I = np.fliplr(I)
    if reflect_y:
        I = np.flipud(I)

    return I
