import numpy as np
import pytest
import cv2
from click.testing import CliRunner
from pathlib import Path
from src import data_gen_pipeline_command
from src.data import (
    make_zero_bg, save_inverted_image, 
    add_grad, add_lines, add_rand_lines, 
    add_noise, img_rotation, scale_factor_img_rotation
    )
import yaml
# from tests import PATH_TO_GRADIENT_IMGS

EPS = 10**(-9)

@pytest.fixture(scope="function")
def seed_zero():
    seed = 0
    np.random.seed(seed)
    return None

def check_zero_bg(tst_bg, bg_size): 
    check = (type(tst_bg) == np.ndarray) \
        and (tst_bg.shape == bg_size) \
            and (np.all(tst_bg == 0))  
    return check

def test_make_zero_bg():
    bg_size = (7, 9)
    tst_bg = make_zero_bg(bg_size)
    assert check_zero_bg(tst_bg, bg_size)


@pytest.mark.parametrize(
    "prob, min_c, max_c, direction, img_name",
    [(1.0, 0.1, 0.9, "horizontal", "h_grad_1.png"),
    (0.1, 0.1, 0.9, "horizontal", "h_grad_2.png"),
    (0.9, 0.1, 0.7, "vertical", "v_grad_1.png"),
    (0.9, 0.1, 0.5, "diagonal", "d_grad_1.png"),
    ]
)
def test_add_grad(prob, min_c, max_c, direction, img_name, seed_zero, path_assert_gradient_imgs):
    bg_size = (100, 100)
    tst_bg = make_zero_bg(bg_size)
    tst_img = add_grad(tst_bg, prob, min_c, max_c, direction)
    tst_img = np.uint8(np.clip((1 - tst_img), 0, 1) * 255)
    img_path = Path(path_assert_gradient_imgs, img_name).as_posix()
    # cv2.imwrite(img_path, tst_img) #, [cv2.IMWRITE_JPEG_QUALITY, 100]
    exmpl_img = np.uint8(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    assert exmpl_img is not None    
    assert np.array_equal(tst_img, exmpl_img)
    assert check_zero_bg(tst_bg, bg_size)

@pytest.mark.parametrize(
        "h, w, angle_rotation, sf_assert", 
        [(1, 1, 0, 1), (11, 11, 0, 1), (5, 5, 90, 1), (12, 12, 180, 1), 
         (7, 7, 270, 1), (12, 12, 360, 1), (7, 7, 450, 1), (12, 12, 540, 1),
         (3, 7, 0, 1), (1, 12, 180, 1), (17, 7, 360, 1), (3, 7, 540, 1),])
def test_simple_scale_factor_img_rotation(h, w, angle_rotation, sf_assert):    
    assert np.abs(scale_factor_img_rotation(h, w, angle_rotation) - sf_assert) < EPS

def test_square_scale_factor_img_rotation():
    sin_ratio_75_45 = np.sin((75 / 180) * np.pi) / np.sin((45 / 180) * np.pi)  
    scf_square_30 = scale_factor_img_rotation(3, 3, 30)  
    scf_square_60 = scale_factor_img_rotation(1000, 1000, 60) 
    scf_square_120 = scale_factor_img_rotation(1000, 1000, 120) 
    scf_square_150 = scale_factor_img_rotation(2000, 2000, 150)
    scf_square_210 = scale_factor_img_rotation(450, 450, 210)
    scf_square_240 = scale_factor_img_rotation(400, 400, 240)
    scf_square_300 = scale_factor_img_rotation(555, 555, 300)
    scf_square_330 = scale_factor_img_rotation(200, 200, 330)
    scf_square_390 = scale_factor_img_rotation(555, 555, 390)
    scf_square_420 = scale_factor_img_rotation(250, 250, 420)
    assert scf_square_30 == scf_square_60 == scf_square_120 == scf_square_150
    assert scf_square_30 == scf_square_210 == scf_square_240 == scf_square_300
    assert scf_square_30 == scf_square_330 == scf_square_390 == scf_square_420
    assert np.abs(scf_square_30 - sin_ratio_75_45) < EPS

    sin_ratio_90_45 = np.sin((90 / 180) * np.pi) / np.sin((45 / 180) * np.pi)
    scf_square_45 = scale_factor_img_rotation(3, 3, 45)  
    scf_square_135 = scale_factor_img_rotation(100, 100, 135) 
    scf_square_225 = scale_factor_img_rotation(2000, 2000, 225) 
    scf_square_315 = scale_factor_img_rotation(777, 777, 315) 
    scf_square_405 = scale_factor_img_rotation(17, 17, 405)
    assert scf_square_45 == scf_square_135 == scf_square_225 == scf_square_315 == scf_square_405
    assert np.abs(scf_square_45 - sin_ratio_90_45) < EPS

def test_rectangle_scale_factor_img_rotation():
    W = 1_000_000_000
    H = round(3**(1/2) * W)
    sin_ratio_120_30 = np.sin((120 / 180) * np.pi) / np.sin((30 / 180) * np.pi)
    scf_rectangle_30 = scale_factor_img_rotation(H, W, 30)
    scf_rectangle_150 = scale_factor_img_rotation(H, W, 150)
    scf_rectangle_210 = scale_factor_img_rotation(H, W, 210)
    scf_rectangle_330 = scale_factor_img_rotation(H, W, 330)
    assert scf_rectangle_30 == scf_rectangle_150 == scf_rectangle_210 == scf_rectangle_330
    assert np.abs(scf_rectangle_30 - sin_ratio_120_30) < EPS

    sin_ratio_90_30 = np.sin((90 / 180) * np.pi) / np.sin((30 / 180) * np.pi)
    scf_rectangle_60 = scale_factor_img_rotation(H, W, 60)
    scf_rectangle_120 = scale_factor_img_rotation(H, W, 120)
    scf_rectangle_240 = scale_factor_img_rotation(H, W, 240)
    scf_rectangle_300 = scale_factor_img_rotation(H, W, 300)
    assert scf_rectangle_60 == scf_rectangle_120 == scf_rectangle_240 == scf_rectangle_300
    assert np.abs(scf_rectangle_60 - sin_ratio_90_30) < EPS

    sin_ratio_60_30 = np.sin((60 / 180) * np.pi) / np.sin((30 / 180) * np.pi)
    scf_rectangle_90 = scale_factor_img_rotation(H, W, 90)
    scf_rectangle_270 = scale_factor_img_rotation(H, W, 270)
    assert scf_rectangle_90 == scf_rectangle_270
    assert np.abs(scf_rectangle_90 - sin_ratio_60_30) < EPS
    assert np.abs(scf_rectangle_60 - sin_ratio_60_30) > EPS
    



def glob_files_names(dir_path, pattern):
    files_names = [file_path.name for file_path in Path(dir_path).glob(pattern)]
    return files_names

@pytest.mark.slow
def test_data_gen_pipeline_command(config_path_and_config_dict, path_assert_imgs_masks):
    config_path, config_dict = config_path_and_config_dict
    runner = CliRunner()
    result = runner.invoke(
        data_gen_pipeline_command, 
        f"--config_path {config_path}")
    assert result.exit_code == 0

    imgs_names = glob_files_names(config_dict["output_img_path"], "*.jpg")
    masks_names = glob_files_names(config_dict["output_mask_path"], "*.jpg")

    ds_config_size = config_dict["ds_size"]
    sub_ds_sizes = config_dict["sub_datasets_size"]
    ds_real_size = sum(
        [round(sub_ds_size * ds_config_size) for sub_ds_size in sub_ds_sizes])

    assert len(imgs_names) == len(masks_names) == ds_real_size

    path_to_assert_imgs, path_to_assert_masks = path_assert_imgs_masks

    assert_imgs_names = glob_files_names(path_to_assert_imgs, "*.jpg")
    assert_masks_names = glob_files_names(path_to_assert_masks, "*.jpg")

    assert assert_imgs_names == imgs_names  
    assert assert_masks_names == masks_names

    for img_name in imgs_names:
        gen_img_path = Path(config_dict["output_img_path"], img_name).as_posix()
        assert_img_path = Path(path_to_assert_imgs, img_name).as_posix()
        gen_img = cv2.imread(gen_img_path, cv2.IMREAD_GRAYSCALE)
        assert_img = cv2.imread(assert_img_path, cv2.IMREAD_GRAYSCALE)

        assert gen_img is not None
        assert np.all(gen_img == assert_img)
    
    for mask_name in masks_names:
        gen_mask_path = Path(config_dict["output_mask_path"], mask_name).as_posix()
        assert_mask_path = Path(path_to_assert_masks, mask_name).as_posix()
        gen_mask = cv2.imread(gen_mask_path, cv2.IMREAD_GRAYSCALE)
        assert_mask = cv2.imread(assert_mask_path, cv2.IMREAD_GRAYSCALE)

        assert gen_mask is not None
        assert np.all(gen_mask == assert_mask)
