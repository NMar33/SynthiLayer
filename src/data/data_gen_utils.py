"""
Utilities for generating synthetic datasets.
"""

import math
from typing import List, Union
import numpy as np
from numpy.random import rand
import cv2

# from entities import DsGenParams


def add_h_v_grad(
    bg_img: np.ndarray, min_c: float, max_c: float, direction: str
) -> np.ndarray:

    min_c, max_c = min_c, min_c + (max_c - min_c) * rand()
    order = np.random.choice([-1, 1])
    h, w = bg_img.shape
    if direction == "horizontal":
        grad = np.linspace(min_c, max_c, h * w)[::order].reshape(w, h).transpose(1, 0)
    elif direction == "vertical":
        grad = np.linspace(min_c, max_c, h * w)[::order].reshape(h, w)

    return bg_img + grad


def add_d_grad(bg_img: np.ndarray, min_c: float, max_c: float) -> np.ndarray:

    max_c = min_c + (max_c - min_c) * rand()
    h, w = bg_img.shape
    shift = np.random.randint(0, 2)
    period = w - 1 + 2 * np.random.rand()
    p, s = period, shift
    grad = np.array(
        [
            min_c
            + (max_c - min_c)
            * ((4 / p) * ((s + i) % p) * (1 - (1 / p) * ((s + i) % p)))
            for i in range(h * w)
        ]
    ).reshape(h, w)

    return bg_img + grad


def make_rand_pattern_line(
    bg_img: np.ndarray, p_deflection: float,
    p_momentum: float, pattern_shift_per_row: float
    ) -> List[int]:

    """
    Produces a template to be used to create lines.

    Parameters:

    bg_img - image

    p_deflection - probability of deflection from a straight line.

    p_momentum - probability that the line will continue to deviate in the same direction.

    pattern_shift_per_row - pattern shift (depends on distance between lines)

    the pattern shifting will be calculated
    as the product of pattern_shift and the distance (in rows) between lines
    """

    img_h, img_w = bg_img.shape
    pattern_line = [0]
    displacement = np.random.choice([-1, 1])
    for _ in range(img_w + math.ceil(pattern_shift_per_row * img_h)):
        p_d, p_m = np.random.rand(2)
        if p_d < p_deflection:
            if p_m > p_momentum:
                displacement = -1 * displacement
            pattern_line.append(displacement)
        else:
            pattern_line.append(0)
    return pattern_line

def add_lines(
    bg_img: np.ndarray, p_deflection: float, p_momentum: float,
    pattern_shift_per_row: float, p_pattern: float,
    p_rupture: float, p_continuation: float, num_lines: int,
    thickness: List[int], dist_between: List[int],
    color_min: float, color_max: float, color_fluctuation: float = 0.2) -> np.ndarray:
    
    """
    Produces a random pattern and adds random lines that
    follow that pattern to the image
    
    Parameters: 

    bg_img - image

    p_deflection - the probability of deflection from a straight line.

    p_momentum - the probability that the line will continue to deviate in the same direction.

    pattern_shift_per_row - pattern shift 
    the pattern shifting will be calculated
    as the product of pattern_shift and the distance (in rows) between lines

    p_pattern - the probability that the added lines will follow the main pattern.

    p_rupture - the probability of a line rupture.

    p_continuation - the probability that the line will continue again after the rupture.

    num_lines - the number of lines to be added to the image.

    thickness - list with line thikness in pixels.

    dist_between - list with line spacings in pixels.

    Returns:

    Modified image
    """

    img_h, img_w = bg_img.shape
    lines_mask = np.zeros_like(bg_img)
    pattern_line = make_rand_pattern_line(bg_img, p_deflection, p_momentum, pattern_shift_per_row)

    start_row = 0
    for line in range(num_lines):
        line_color = color_min + (color_max - color_min) * rand()
        start_row = start_row + dist_between[line]
        if start_row >= img_h:
            break
        pattern_shift = round(start_row * pattern_shift_per_row)
        r_new = start_row
        r_prev = r_new
        if p_rupture != 0:
            p_initial_rupture = np.array([p_rupture, p_continuation])
            p_initial_rupture = p_initial_rupture / np.sum(p_initial_rupture)
        else:
            p_initial_rupture = np.array([0, 1])
        rupture = np.random.choice([True, False], p=p_initial_rupture)
        for col in range(img_w):
            p_rup, p_cont, p_patt = rand(3)
            if rupture:
                rupture = ~(p_cont < p_continuation)
            else:
                rupture = p_rup < p_rupture
            if p_patt < p_pattern:
                r_new += pattern_line[col + pattern_shift]
            r_min = max(min(r_prev, r_new), 0)
            r_max = min(max(r_prev, r_new) + thickness[line], img_h - 1)
            if (r_min < img_h) and (0 <= r_max) and ~rupture:
                lines_mask[r_min : r_max, col] = line_color * (
                    (1 + 2 * color_fluctuation) 
                    - color_fluctuation * rand(r_max - r_min))
            r_prev = r_new
        start_row = start_row + thickness[line]
    return bg_img + lines_mask

def get_random_parameters_list(
    parameter_min_max: Union[List[int], List[float]], 
    parameter_size: int, 
    ) -> Union[List[int], List[float]]:
    parameter_min, parameter_max = parameter_min_max
    if isinstance(parameter_min, int) and isinstance(parameter_max, int):
        param = np.random.randint(parameter_min, parameter_max + 1, parameter_size)
    elif isinstance(parameter_min, float) and isinstance(parameter_max, float):
        param = parameter_min + (parameter_max - parameter_min) * rand(parameter_size)
    else:
        raise Exception(f"Wrong Configs. {parameter_min_max}, must be List[int, int] or List[float, float]")
    return list(param)

def get_random_parameter(
    parameter_min_max: Union[List[int], List[float]]
    ) -> Union[int, float]:
    param = get_random_parameters_list(parameter_min_max, 1)[0]
    return param



# print("data gen utils")
