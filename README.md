# SynthiLayer
## Synthetic Dataset Generator for Layered Structures

Hello :smiley:, thank you for being here! SynthiLayer is a program designed to generate images that mimic layered structures. Additionally, it produces masks to highlight recurring patterns. 

Synthetic datasets produced by SynthiLayer are suitable for pre-training different segmentation models. They may prove beneficial when there's a need to segment photographs or micrographs of the following layered objects:

- Graphene oxide membranes
- Mica
- Stromatolites
- Shale rocks
- Sedimentary rocks
- Plywood
- Laminated glass
- Layered silicates
- Archaeological strata
- Ice core layers
- Laminated fabrics and textiles
- Onion skin layers
- Lake or ocean sediment layers
- Tree rings (for dendrochronology)
- Striated muscle tissue
- Metal composites with layered structures
- Nacre (also known as mother of pearl)
- Layered ceramics
- Schist
- Varves
- Techniques like layered chromatography
- Layered solar cells

...and yes, even the delicious layers of **Tiramisu**!

## Installation

1. Clone the repository or download the source code.
2. Create a virtual environment
```bash
python -m venv .venv
```
3. Activate the virtual environment:

- Windows:
  ```bash
  .venv\Scripts\activate.bat
  ```

- Linux/Mac:
  ```bash
  source .venv/bin/activate
  ```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

Let's make the README more streamlined and cohesive:

## How to Use

1. Activate your virtual environment (if it's not already active).
2. If needed, modify the primary configuration file: 
   ```
   configs\data_gen_config.yaml
   ```
   Also, update the sub-dataset configuration files as required:
   ```
   configs\ds_type_01_500-01.yaml
   ...
   ```
3. Execute the script by running the following command:
```bash
python src/data_gen_pipeline.py --config_path configs/data_gen_config.yaml
```

## Configuration File Explanation
The primary parameters are specified in the `configs/data_gen_config.yaml` file. These parameters determine aspects like the number of images to be generated (`ds_size`). It also defines the path to the sub-dataset configuration files (`sub_datasets_config_path`) and their sizes (`sub_datasets_size`).

### Main Config Example
```yaml
# main config example
logging_config_path: "configs/data_gen_logging_config.yaml" # Either specify the logging_config_path or use "default"

random_state: -1 # If you don't wish to specify a random seed, use -1 for this field
ds_size: 150
output_img_path: "draft_img"
output_mask_path: "draft_mask"
sub_datasets_size: 
  - 0.33
  - 0.33
  - 0.33

sub_datasets_config_path:
  - "configs/ds_type_01_500-01.yaml"
  - "configs/ds_type_02_500-01.yaml"
  - "configs/ds_type_03_500-01.yaml"
```
Using this configuration file, 150 images will be generated. These images will be divided into three sub-datasets, with each sub-dataset containing 50 images.

### Sub-dataset Config Example
```yaml
# sub_dataset config example
img_size: [500, 500]
img_name_prefix: "type_01_500"

noise_pipeline: [
  ["add_grad", {prob: 0.5, min_c: 0.01, max_c: 0.1, direction: "horizontal"}],
  ["add_grad", {prob: 0.5, min_c: 0.01, max_c: 0.1, direction: "vertical"}],
  ["add_grad", {prob: 0.5, min_c: 0.01, max_c: 0.1, direction: "diagonal"}],

  ["add_noise", {prob: 0.5, min_c: 0.01, max_c: 0.1}],

]

# ...
```
When using this sub_dataset config file, images will be sized at 500x500 pixels and their names will start with "type_01_500". 

Noise elements are defined by the `noise_pipeline` parameter and will include gradients (horizontal, vertical, and diagonal; gradient examples can be viewed in the `tests\test_images\gradients` folder) as well as Gaussian noise. The likelihood of adding a gradient or noise to the image is determined by the `prob` parameter. The `min_c` and `max_c` parameters define the minimum and maximum color values of the gradient or noise. An exact value is randomly selected from the range [min_c, max_c]. It's important to note that a value of 1 corresponds to the color black, and 0 corresponds to white. Hence, the larger the values of min_c and max_c, the darker the generated gradient will be.

```yaml
# sub_dataset config example continued
# ...
mask_pipeline: [
  ["add_rand_lines", {
    num_lines_min_max: [5, 70],
    thickness_min_max: [1, 18],
    dist_between_min_max: [5, 25],
    p_rupture_min_max: [0.0, 0.01],
    p_continuation_min_max: [0.0, 0.02],
    color_min_max: [0.01, 0.4],
    p_deflection_min_max: [0.01, 0.4], 
    p_momentum_min_max: [0.7, 0.99],
    pattern_shift_per_row_min_max: [0.0, 2.0],
    p_pattern_min_max: [0.6, 0.99],
  }],
]
```

Elements that should be targeted during the segmentation process are defined by the mask_pipeline parameter, which contains the sequence of steps required for generating the target elements of the image and its mask. In this example, only one step is used - `add_rand_lines`. This step produces a random pattern and adds random lines that follow this pattern to the image. The parameters within this step define various characteristics of the lines, such as the number of lines, their thickness, distance between them, and other aspects that influence their appearance and placement.

#### Parameters for Line Generation

The parameters `num_lines_min_max: [5, 70]`, `thickness_min_max: [1, 18]`, and `dist_between_min_max: [5, 25]` control the number of lines to be generated, their thickness, and the distance between them. In this example, between 5 to 17 lines will be generated, with a thickness ranging from 1 to 18 pixels. The distance between these lines will vary from 5 to 25 pixels. It's important to note that the first line will be positioned at a distance of 5 to 15 pixels from the top edge of the image, as the `dist_between_min_max: [5, 25]` parameter also determines the placement of the first line.

The `p_rupture_min_max` parameter sets the likelihood that a line will break, while the `p_continuation_min_max` parameter sets the probability that a line will continue after a break. In this current example with `p_rupture_min_max: [0.0, 0.01]` and `p_continuation_min_max: [0.0, 0.02]`, for each generated image, the values for `p_rupture` and `p_continuation` will be randomly chosen from the ranges [0.0, 0.01] and [0.0, 0.02], respectively. Subsequently, during the generation of lines on the respective image, these chosen values for `p_rupture` and `p_continuation` will be applied.

#### Parameters for Pattern Generation

`p_deflection_min_max` - [min, max] represents the probability of a main pattern deflecting from being straight. If this parameter is set to zero, the main pattern will be generated as a straight line.

`p_momentum_min_max` - [min, max] indicates the likelihood that once a pattern has deviated from being straight, it will continue to deviate in the same direction. With a lower value for this parameter, the wavy nature of the generated pattern will also be lower. However, if the value is very high, the pattern will stochastically oscillate around a straight trajectory.

`pattern_shift_per_row_min_max` - [min, max] denotes the shift in the pattern, where the shift will be determined as the product of the pattern shift and the distance (in rows) between lines. When this parameter is zero, one line will be positioned directly under another, repeating the pattern on every image row. If this parameter has a high value, the pattern will shift on every image row, creating an effect of offset lines.

`p_pattern_min_max` - [min, max] is the probability that the added lines will follow the primary pattern. If this parameter is zero, lines will be generated at random. However, with a higher value, lines will be generated in accordance with the pattern.
