import os
import logging
import yaml
import numpy as np
from typing import List
from pathlib import Path

import click
from tqdm import tqdm

from entities import (
    DsGenParams,
    SubDsGenParams,
    read_ds_gen_params,
    read_sub_ds_gen_params,
)

from data import (
    make_zero_bg, 
    add_grad, 
    save_inverted_image, 
    add_lines, 
    add_rand_lines, 
    add_noise,
    img_rotation)

from utils import check_make_dirs, setup_logging

GENERATION_METHODS = {
    "add_grad": add_grad, 
    "add_lines": add_lines, 
    "add_rand_lines": add_rand_lines, 
    "add_noise": add_noise,
    "img_rotation": img_rotation
    }
 

def gen_img_layers(bg_img: np.ndarray, gen_layer_pipeline_config: List) -> np.ndarray:
    """
    Generate image layers based on a background image and configuration parameters.
    
    This function applies a series of processing methods to the background image 
    according to the provided configuration. It can be used for generating masks, 
    adding noise, and other image modifications based on configuration parameters.

    Parameters:
    ----------
    bg_img : np.ndarray
        The background image on which the layers will be generated.
    gen_layer_pipeline_config : List
        A list of tuples where each tuple contains a processing method and its parameters.

    Returns:
    -------
    np.ndarray
        The processed image after applying all methods.

    Raises:
    ------
    Exception
        If method parameters are not a tuple, list, or dictionary.

    Example:
    -------
    bg = make_zero_bg(sub_ds_gen_params.img_size)
    mask = gen_img_layers(bg, sub_ds_gen_params.mask_pipeline)
    noise = gen_img_layers(bg, sub_ds_gen_params.noise_pipeline)
    img = noise + mask
    """
    for method, method_params in gen_layer_pipeline_config:
        if isinstance(method_params, (tuple, list)):
            bg_img = GENERATION_METHODS[method](bg_img, *method_params)
        elif isinstance(method_params, dict):
            bg_img = GENERATION_METHODS[method](bg_img, **method_params)
        else:
            raise Exception(f"Wrong Configs. Method parameters must be a tuple, list, or dictionary. Current type: {type(method_params)}")
    return bg_img

def save_path(base_path: str, prefix: str, index: int) -> str:
    "Formulate the path for saving the image/mask."
    return Path(base_path, f"{prefix}_{index:03d}.jpg").as_posix()

def data_gen_pipeline(ds_gen_params: DsGenParams) -> None:
    "Manage the entire data generation process."
    logger = logging.getLogger("data_gen." + __name__)
    logger.info("start data_gen_pipeline with params %s", ds_gen_params)
    
    SEED = ds_gen_params.random_state
    if SEED != -1:
        np.random.seed(SEED)

    check_make_dirs([
        ds_gen_params.output_img_path, 
        ds_gen_params.output_mask_path
        ])
    
    ds_size = ds_gen_params.ds_size
    sub_datasets_size = [
        round(sub_ds_size * ds_size) 
        for sub_ds_size in ds_gen_params.sub_datasets_size]

    for sub_ds_size, sub_ds_config_path in zip(
        sub_datasets_size, ds_gen_params.sub_datasets_config_path):
        
        sub_ds_gen_params = read_sub_ds_gen_params(sub_ds_config_path)
        logger.debug("Start data_set generation. Size: %s. Parameters: %s", sub_ds_size, sub_ds_gen_params)
        
        for i in tqdm(range(sub_ds_size), sub_ds_config_path):
            bg = make_zero_bg(sub_ds_gen_params.img_size)
            mask = gen_img_layers(bg, sub_ds_gen_params.mask_pipeline)
            noise = gen_img_layers(bg, sub_ds_gen_params.noise_pipeline)
            img = noise + mask
            img = np.clip(img, 0, 1)
            mask[mask > 0] = 1
            mask = np.clip(mask - (noise == 1), 0, 1)
            
            output_img_path = save_path(ds_gen_params.output_img_path, sub_ds_gen_params.img_name_prefix, i+1)
            output_mask_path = save_path(ds_gen_params.output_mask_path, sub_ds_gen_params.img_name_prefix, i+1)
            
            save_inverted_image(mask, output_mask_path)
            save_inverted_image(img, output_img_path)


@click.command(
    name="data_gen_pipeline",
    help="""The program generates images with a layered structure
    and a masks of a layered structure for subsequent
    training of the neural network.""",
)
@click.option(
    "--config_path",
    type=str,
    prompt="Please, enter the config file path.",
    help="Specify the path to the dataset generation config file.",
)
def data_gen_pipeline_command(config_path: str) -> None:
    "Command line interface for the data generation pipeline."
    ds_gen_params = read_ds_gen_params(config_path)
    setup_logging(ds_gen_params.logging_config_path)
    data_gen_pipeline(ds_gen_params)


if __name__ == "__main__":
    data_gen_pipeline_command()
