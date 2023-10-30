"""
Scripts for creating dataclasses.
"""

import sys

from .ds_gen_params import (
    DsGenParams,
    SubDsGenParams,
    read_ds_gen_params,
    read_sub_ds_gen_params,
)

__all__ = [
    "DsGenParams",
    "SubDsGenParams",
    "read_ds_gen_params",
    "read_sub_ds_gen_params",
]
