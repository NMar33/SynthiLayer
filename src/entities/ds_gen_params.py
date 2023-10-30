"""
Dataclasses for synthetic dataset generation.
"""
# check annotation

from dataclasses import dataclass
from typing import Tuple, List, Any
from marshmallow_dataclass import class_schema


import yaml


@dataclass()
class DsGenParams:
    logging_config_path: str
    random_state: int
    ds_size: int
    output_img_path: str
    output_mask_path: str
    sub_datasets_size: List[float]
    sub_datasets_config_path: List[str]


@dataclass()
class SubDsGenParams:
    """
    Dataclass to store parameters for generating a subset of dataset images.
    
    The parameters encapsulate image properties, naming conventions, and 
    processing pipelines for adding noise and creating masks.
    
    Attributes:
    ----------
    img_size : Tuple[int, int]
        Dimensions of the generated image as (height, width).
    
    img_name_prefix : str
        Prefix for generated image filenames.
    
    noise_pipeline : List[Tuple[str, dict]]
        Sequence of operations (and their parameters) to introduce noise into the image.
        Each operation is represented as a tuple where the first element is a string 
        indicating the operation name, and the second element is a dictionary of parameters 
        for that operation.

    mask_pipeline : List[Tuple[str, dict]]
        Sequence of operations (and their parameters) for mask creation and modification.
        The structure is similar to `noise_pipeline`.
    """

    img_size: Tuple[int, int]
    img_name_prefix: str
    noise_pipeline: List[Tuple[str, dict]]
    mask_pipeline: List[Tuple[str, dict]]


DsGenParamsSchema = class_schema(DsGenParams)
SubDsGenParamsSchema = class_schema(SubDsGenParams)


def read_ds_gen_params(path: str) -> DsGenParams:
    with open(path, "r") as input_stream:
        schema = DsGenParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def read_sub_ds_gen_params(path: str) -> SubDsGenParams:
    with open(path, "r") as input_stream:
        schema = SubDsGenParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
