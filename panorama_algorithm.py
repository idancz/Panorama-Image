"""Projective Homography and Panorama Algorithms."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata

import matplotlib.pylab as plt
import cv2

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Algorithms:
    """Implement Projective Homography and Panorama Algorithms."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        A = []
        for i in range(match_p_src.shape[1]):
            u_src, v_src = match_p_src[0, i], match_p_src[1, i]
            u_dst, v_dst = match_p_dst[0, i], match_p_dst[1, i]
            A.append([u_src, v_src, 1, 0, 0, 0, -u_src*u_dst, -u_dst*v_src, -u_dst])
            A.append([0, 0, 0, u_src, v_src, 1, -v_dst*u_src, -v_src*v_dst, - v_dst])
        A = np.asarray(A)
        U, S, Vt = svd(A)  # Vt - matrix of eigenvectors
        homography = Vt[-1].reshape(3, 3)
        return homography

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        dst_image = np.zeros(shape=dst_image_shape, dtype=np.uint8)
        for y in range(src_image.shape[0]):
            for x in range(src_image.shape[1]):
                X_src = np.array([x, y, 1]).transpose()
                X_dst = homography @ X_src
                corr = np.array([X_dst[0] / X_dst[2], X_dst[1] / X_dst[2]])
                corr = corr.round().astype(int)
                if(0 <= corr[0] < dst_image_shape[1]) and (0 <= corr[1] < dst_image_shape[0]):
                    dst_image[corr[1], corr[0]] = src_image[y, x]
        return dst_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        new_image = np.zeros(shape=dst_image_shape, dtype=np.uint8)
        hw_length = src_image.shape[0] * src_image.shape[1]
        x = np.linspace(0, src_image.shape[0]-1, src_image.shape[0]).astype(int)
        y = np.linspace(0, src_image.shape[1]-1, src_image.shape[1]).astype(int)
        yy, xx = np.meshgrid(y, x)
        yy = yy.reshape((1, hw_length))
        xx = xx.reshape((1, hw_length))
        ones = np.ones(shape=(1, hw_length))
        X = np.concatenate((yy, xx, ones), axis=0)
        Y = homography @ X
        Y /= Y[-1]
        Y_norm = Y[0:2]
        Y_norm = Y_norm.round().astype(int)

        mask = (Y_norm[1] >= 0) & (Y_norm[1] < dst_image_shape[0]) & (Y_norm[0] >= 0) & (Y_norm[0] < dst_image_shape[1])
        new_image[Y_norm[1, mask], Y_norm[0, mask]] = src_image[xx[0, mask], yy[0, mask]]
        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        mse = []
        cnt = 0
        for i in range(match_p_src.shape[1]):
             u_src, v_src = match_p_src[0, i], match_p_src[1, i]
             u_dst, v_dst = match_p_dst[0, i], match_p_dst[1, i]
             X = np.array([u_src, v_src, 1]).transpose()
             Y = homography @ X
             corr = np.array([Y[0] / Y[2], Y[1] / Y[2]])
             dist = np.sqrt((corr[0] - u_dst)**2 + (corr[1] - v_dst)**2)
             if dist <= max_err:
                cnt += 1
                mse.append(dist**2)
        fit_percent = (cnt / match_p_src.shape[1])
        if not mse:
            return tuple([fit_percent, 10 ** 9])
        dist_mse = np.average(mse)
        return tuple([fit_percent, dist_mse])



    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        mp_src_meets_model, mp_dst_meets_model = [[], []], [[], []]
        for i in range(match_p_src.shape[1]):
            u_src, v_src = match_p_src[0, i], match_p_src[1, i]
            u_dst, v_dst = match_p_dst[0, i], match_p_dst[1, i]
            X = np.array([u_src, v_src, 1]).transpose()
            Y = homography @ X
            corr = np.array([Y[0] / Y[2], Y[1] / Y[2]])
            dist = np.sqrt((corr[0] - u_dst) ** 2 + (corr[1] - v_dst) ** 2)
            if dist <= max_err:
                mp_src_meets_model[0].append(u_src)
                mp_src_meets_model[1].append(v_src)
                mp_dst_meets_model[0].append(u_dst)
                mp_dst_meets_model[1].append(v_dst)
        return np.asarray(mp_src_meets_model), np.asarray(mp_dst_meets_model)

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        d *= match_p_src.shape[1]
        mse = 10 ** 9
        homography = None
        for _ in range(k):
            random_idx = np.random.choice(match_p_src.shape[1], n, replace=False)
            temp_src_p = match_p_src[:, random_idx]
            temp_dst_p = match_p_dst[:, random_idx]
            random_homography = Algorithms.compute_homography_naive(temp_src_p, temp_dst_p)
            new_src_p, new_dst_p = Algorithms.meet_the_model_points(random_homography, match_p_src, match_p_dst, t)
            if new_src_p.shape[1] > d:
                inliers_homography = Algorithms.compute_homography_naive(new_src_p, new_dst_p)
                temp_percentage, temp_mse = Algorithms.test_homography(inliers_homography, match_p_src, match_p_dst, t)
                if temp_mse < mse:
                    mse = temp_mse
                    homography = inliers_homography
        try:
            assert homography is not None, "RANSAC did not converge, please run again!"  # RANSAC failed
        except AssertionError as e:
            print(e)
            exit(1)
        return homography


    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        hw_length = dst_image_shape[0] * dst_image_shape[1]
        x = np.linspace(0, dst_image_shape[0]-1, dst_image_shape[0]).astype(int)
        y = np.linspace(0, dst_image_shape[1]-1, dst_image_shape[1]).astype(int)
        xx, yy = np.meshgrid(x, y)
        yy = yy.reshape((1, hw_length))
        xx = xx.reshape((1, hw_length))
        ones = np.ones(shape=(1, hw_length))
        X_dst = np.concatenate((yy, xx, ones), axis=0)
        Y_src = backward_projective_homography @ X_dst
        Y_src /= Y_src[-1]
        Y_norm = Y_src[0:2]
        backward_warp = np.zeros(shape=dst_image_shape, dtype=np.uint8)
        Y_rounded = np.ceil(Y_norm).astype(int)
        mask = (Y_rounded[1] >= 0) & (Y_rounded[1] < (src_image.shape[0])) & (Y_rounded[0] >= 0) & (Y_rounded[0] < (src_image.shape[1]))
        h, w, grid_h, grid_w = Y_rounded[1, mask], Y_rounded[0, mask], Y_norm[1, mask], Y_norm[0, mask]
        pixels = griddata((h, w), src_image[h, w], (grid_h, grid_w), method='linear')
        backward_warp[xx[0, mask], yy[0, mask]] = pixels
        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        translation = np.matrix([[1, 0, -pad_left], [0, 1, -pad_up], [0, 0, 1]])
        final_homography = backward_homography @ translation
        final_homography /= np.linalg.norm(final_homography)
        return np.asarray(final_homography).reshape(3, 3)

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        solution = Algorithms()
        forward_homography = solution.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err) # (1.a)
        panorama_rows_num, panorama_cols_num, pad_struct = Algorithms.find_panorama_shape(src_image, dst_image,forward_homography) # (1.b)
        backward_homography = np.linalg.inv(forward_homography)
        backward_homography /= np.linalg.norm(backward_homography)
        panorama_homography = Algorithms.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left, pad_struct.pad_up) # (3)
        panorama = np.zeros(shape=(panorama_rows_num, panorama_cols_num, 3), dtype=np.uint8)  # (5)
        backward_warped = Algorithms.compute_backward_mapping(panorama_homography, src_image, panorama.shape)  # (4)
        h, w = pad_struct.pad_up, pad_struct.pad_left
        panorama[h:h+dst_image.shape[0], w:w+dst_image.shape[1]] = dst_image
        u, v = backward_warped.shape[0], backward_warped.shape[1]
        mask = panorama[:u, :v] == [0, 0, 0]
        panorama[mask] = backward_warped[mask]
        return np.clip(panorama, 0, 255)
