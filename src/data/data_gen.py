"""
Main functions for generating a synthetic dataset.
"""

import numpy as np
from numpy.random import rand
import cv2
from typing import Tuple, List, Union

from .data_gen_utils import add_h_v_grad, add_d_grad, add_lines, get_random_parameter, get_random_parameters_list

# from entities import DsGenParams


def make_zero_bg(size: Tuple[int, int]) -> np.ndarray:
    "Create a black image of the specified size."
    height, width = size
    return np.zeros((height, width))


def add_grad(
    bg_img: np.ndarray, 
    prob: float, 
    min_c: float, 
    max_c: float, 
    direction: str
    ) -> np.ndarray:
    """
    Add a gradient to the input image based on the specified parameters.
    
    Parameters:
    ----------
    bg_img : np.ndarray
        Input image to which the gradient will be added.
    
    prob : float
        Probability of adding a gradient to the image.
    
    min_c : float
        Minimum gradient intensity.
    
    max_c : float
        Maximum gradient intensity. The actual intensity is determined by:
        max_c = min_c + (max_c - min_c) * rand()
    
    direction : str
        Direction of the gradient. Can be one of "horizontal", "vertical", or "diagonal".
    
    Returns:
    -------
    np.ndarray
        Image with the added gradient.
    """

    if rand() < prob:
        if direction in {"horizontal", "vertical"}:
            bg_img = add_h_v_grad(bg_img, min_c, max_c, direction)
        elif direction == "diagonal":
            bg_img = add_d_grad(bg_img, min_c, max_c)

    return bg_img

def add_noise(
    bg_img: np.ndarray, prob: float, min_c: float, max_c: float
    ) -> np.ndarray:
    """
    Add noise to the input image based on specified parameters.
    
    Parameters:
    ----------
    bg_img : np.ndarray
        Input image to which the noise will be added.
    
    prob : float
        Probability of adding noise to the image.
    
    min_c : float
        Minimum noise intensity.
    
    max_c : float
        Maximum noise intensity. The actual intensity is determined by:
        max_c = min_c + (max_c - min_c) * rand()
    
    Returns:
    -------
    np.ndarray
        Image with the added noise, if the probability condition is met; 
        otherwise, returns the original image.
    """

    if rand() < prob:
        h, w = bg_img.shape
        noise = min_c + (max_c - min_c) * np.random.rand(h * w).reshape(h, w)
        return bg_img + noise
    else:
        return bg_img



def add_rand_lines(
    bg_img: np.ndarray, 
    p_deflection_min_max: List[float], 
    p_momentum_min_max: List[float],
    pattern_shift_per_row_min_max: List[float], 
    p_pattern_min_max: List[float],
    p_rupture_min_max: List[float], 
    p_continuation_min_max: List[float], 
    num_lines_min_max: List[int], 
    thickness_min_max: List[int],
    dist_between_min_max: List[int],
    color_min_max: List[float], 
    color_fluctuation: float = 0.2
    ) -> np.ndarray:
    
    """
    Produces a random pattern and adds random lines that
    follow that pattern to the image
    
    Parameters: 

    bg_img - image

    p_deflection_min_max - [min, max] probability of deflection from a straight line.

    p_momentum_min_max - [min, max] probability that the line will continue to deviate in the same direction.

    pattern_shift_per_row_min_max - [min, max] pattern shift 
    the pattern shifting will be calculated
    as the product of pattern_shift and the distance (in rows) between lines

    p_pattern_min_max - [min, max] probability that the added lines will follow the main pattern.

    p_rupture_min_max - [min, max] probability of a line rupture.

    p_continuation_min_max - [min, max] probability that the line will continue again after the rupture.

    num_lines_min_max - [min, max] number of lines to be added to the image.

    thickness_min_max - [min, max] list with line thikness in pixels.

    dist_between_min_max - [min, max] list with line spacings in pixels.

    Returns:

    Modified image
    """

    p_deflection = get_random_parameter(p_deflection_min_max)
    p_momentum = get_random_parameter(p_momentum_min_max)
    pattern_shift_per_row = get_random_parameter(pattern_shift_per_row_min_max)
    p_pattern = get_random_parameter(p_pattern_min_max)
    p_rupture = get_random_parameter(p_rupture_min_max)
    p_continuation = get_random_parameter(p_continuation_min_max)

    num_lines: int = get_random_parameter(num_lines_min_max)

    thickness: List[int] = get_random_parameters_list(thickness_min_max, num_lines)
    dist_between: List[int] = get_random_parameters_list(dist_between_min_max, num_lines)

    color_min, color_max = color_min_max

    bg_img = add_lines(bg_img, p_deflection, p_momentum,
    pattern_shift_per_row, p_pattern,
    p_rupture, p_continuation, num_lines,
    thickness, dist_between,
    color_min, color_max, color_fluctuation)

    return bg_img 

def scale_factor_diag_angle(
    angle_diag_side: Union[int, float], angle_rotation: Union[int, float]) -> float:
    """
    angle_diag_side - angle between the side of the rectangle and the diagonal in degrees
    angle_rotation - rotation angle in degrees
    """
    sin_angle_diag_side = np.sin((angle_diag_side / 180) * np.pi)
    angle_btw_diagonals = 180 - 2 * angle_diag_side
    theta_1 = 180 - (angle_diag_side + angle_rotation % 180)
    theta_2 = 180 - (angle_diag_side + (angle_btw_diagonals + angle_rotation % 180) % 180)
    scale_factor = 1.
    for theta in (theta_1, theta_2):
        if theta > 0:
            scale_factor = max(scale_factor, np.sin((theta / 180) * np.pi) / sin_angle_diag_side)

    return scale_factor


def scale_factor_img_rotation(h, w, angle_rotation):
    """
    h - height
    w - width
    angle_rotation - rotation angle in degrees
    """
    angle_diag_side_1 = (np.arctan(h / w) / np.pi) * 180
    angle_diag_side_2 = 90 - angle_diag_side_1

    scale_factor = 1.
    for angle_diag_side in (angle_diag_side_1, angle_diag_side_2):
        scale_factor = max(scale_factor, scale_factor_diag_angle(angle_diag_side, angle_rotation))
    return scale_factor


def img_rotation(
    bg_img: np.ndarray, prob: float, 
    min_angle: int = 0, max_angle: int = 180, scale: bool = False
    ) -> np.ndarray:
    """
    prob - probability to rotate an image

    min_angle - minimum angle (degrees) of rotation, default = 0 degrees  

    max_angle - maximum angle (degrees) of rotation, default = 0 degrees
    """
    
    if rand() < prob:
        h, w = bg_img.shape
        rotation_center = (int(h // 2), int(w // 2))
        angle = get_random_parameter([min_angle, max_angle])
        scale_factor = 1.
        if scale:
            scale_factor = scale_factor_img_rotation(h, w, angle)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, scale_factor)
        img_rotated = cv2.warpAffine(bg_img, rotation_matrix, (w, h))
        return img_rotated
    else:
        return bg_img


def save_inverted_image(bg_img: np.ndarray, img_path: str) -> None:
    cv2.imwrite(img_path, (1 - bg_img) * 255)


