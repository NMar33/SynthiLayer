from textwrap import dedent
from pathlib import Path
import pytest
import yaml


@pytest.fixture(scope="function")
def path_assert_gradient_imgs():
    PATH_TO_TST_GRADIENT_IMGS = "tests/test_images/gradients"
    return PATH_TO_TST_GRADIENT_IMGS

@pytest.fixture(scope="function")
def path_assert_imgs_masks():
    PATH_TO_TST_IMGS = "tests/test_images/imgs"
    PATH_TO_TST_MASKS = "tests/test_images/masks"
    return PATH_TO_TST_IMGS, PATH_TO_TST_MASKS

@pytest.fixture(scope="function")
def config_ds_gen_dict(tmp_path):
    config_str = dedent(
        """
        logging_config_path: "default"
        random_state: 1 
        ds_size: 10
        output_img_path:
        output_mask_path:
        sub_datasets_size:
        - 0.5
        - 0.5
        sub_datasets_config_path:
        """
        )
        
    config_dict = yaml.safe_load(config_str)

    config_dict["output_img_path"] = Path(tmp_path, "imgs").as_posix()
    config_dict["output_mask_path"] = Path(tmp_path, "masks").as_posix()
    config_dict["sub_datasets_config_path"] = [
        Path(tmp_path, "sub_ds_1_config.yaml").as_posix(),
        Path(tmp_path, "sub_ds_2_config.yaml").as_posix()
        ]

    return config_dict

@pytest.fixture(scope="function")
def config_sub_ds_1():
    config_str = dedent(
        """
        img_size: [100, 100]
        img_name_prefix: "sub_ds_1"
        noise_pipeline: [
        ["add_grad", {prob: 0.9, min_c: 0.1, max_c: 0.3, direction: "diagonal"}],
        ["img_rotation", {prob: 0.9}],
        ["add_grad", {prob: 0.9, min_c: 0.1, max_c: 0.3, direction: "vertical"}],
        ["add_grad", {prob: 0.9, min_c: 0.1, max_c: 0.3, direction: "horizontal"}], 
        ["add_noise", {prob: 0.9, min_c: 0.1, max_c: 0.3}],
        ["img_rotation", {prob: 0.9}]
        ]

        mask_pipeline: [
        ["add_rand_lines", {
            p_deflection_min_max: [0.1, 0.9], 
            p_momentum_min_max: [0.1, 0.9],
            pattern_shift_per_row_min_max: [0.01, 0.2],
            p_pattern_min_max: [0.1, 0.9],
            p_rupture_min_max: [0.1, 0.9],
            p_continuation_min_max: [0.1, 0.9],
            num_lines_min_max: [1, 20],
            thickness_min_max: [1, 20],
            dist_between_min_max: [1, 20],
            color_min_max: [0.1, 0.4]
        }],
        ["img_rotation", {prob: 0.9}]
        ]
        """
        )
    
    config_dict = yaml.safe_load(config_str)
    return config_dict

@pytest.fixture(scope="function")
def config_sub_ds_2():
    config_str = dedent(
        """
        img_size: [100, 100]
        img_name_prefix: "sub_ds_2"
        noise_pipeline: [
        ["add_grad", {prob: 0.9, min_c: 0.1, max_c: 0.3, direction: "diagonal"}],
        ["img_rotation", {prob: 0.9}],
        ["add_grad", {prob: 0.9, min_c: 0.1, max_c: 0.3, direction: "vertical"}],
        ["add_grad", {prob: 0.9, min_c: 0.1, max_c: 0.3, direction: "horizontal"}], 
        ["add_noise", {prob: 0.9, min_c: 0.1, max_c: 0.3}],
        ["img_rotation", {prob: 0.9}]
        ]

        mask_pipeline: [
        ["add_rand_lines", {
            p_deflection_min_max: [0.1, 0.9], 
            p_momentum_min_max: [0.1, 0.9],
            pattern_shift_per_row_min_max: [0.01, 0.2],
            p_pattern_min_max: [0.1, 0.9],
            p_rupture_min_max: [0.1, 0.9],
            p_continuation_min_max: [0.1, 0.9],
            num_lines_min_max: [1, 20],
            thickness_min_max: [1, 20],
            dist_between_min_max: [1, 20],
            color_min_max: [0.1, 0.4]
        }],
        ["img_rotation", {prob: 0.9}]
        ]
        """
        )
    
    config_dict = yaml.safe_load(config_str)
    return config_dict

@pytest.fixture(scope="function")
def config_path_and_config_dict(tmp_path, config_ds_gen_dict, config_sub_ds_1, config_sub_ds_2):
    DS_GEN_CONFIG_PATH = Path(tmp_path, "config_ds_gen_dict.yaml").as_posix()
    with open(DS_GEN_CONFIG_PATH, "w") as f:
        yaml.safe_dump(config_ds_gen_dict, f)
    
    SUB_DS = [config_sub_ds_1, config_sub_ds_2]
    SUB_DS_CONFIG_PATHS = config_ds_gen_dict["sub_datasets_config_path"]
    for sub_ds, sub_ds_conf_p in zip(SUB_DS, SUB_DS_CONFIG_PATHS):
        with open(sub_ds_conf_p, "w") as f:
            yaml.safe_dump(sub_ds, f)
    return DS_GEN_CONFIG_PATH, config_ds_gen_dict
    
