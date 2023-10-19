# DeepSeeColor

DeepSeeColor combines a state-of-the-art underwater image formation model with the computational efficiency of deep learning frameworks. This enables efficient color correction for underwater imagery _in situ_ onboard autonomous underwater vehicles.

For more details, please refer to our paper: 
- Paper: https://ieeexplore.ieee.org/document/10160477
- Project Page: https://warp.whoi.edu/deepseecolor/
- arXiv: https://arxiv.org/abs/2303.04025

If you use DeepSeeColor in your work, please cite it as follows:
> S. Jamieson, J. P. How and Y. Girdhar, "DeepSeeColor: Realtime Adaptive Color Correction for Autonomous Underwater Vehicles via Deep Learning Methods," 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 3095-3101, doi: 10.1109/ICRA48891.2023.10160477.

## Dependencies

- Python3 (v3.8 or later recommended)
- Pytorch (v2.0 or later recommended)
- Torchvision
- Kornia

An `environment.yaml` file is provided for conda installations, which can be configured by running
```bash
conda env create -f environment.yaml  # Tested on Ubuntu 20.04
```

## Usage

DeepSeeColor requires two inputs:
1. A directory of RGB (or BGR) images
2. A directory of corresponding single-channel depth images, where depth is encoded as either a 32-bit floating point value representing metres, or an unsigned 16-bit integer representing millimetres.

Currently, it expects that each directory contains the same number of files and that they have matching lexicograhical order. For an example, see the linked datsets at the end of this readme.

Run `python3 deepseecolor.py --help` for CLI usage.

## Datasets

The authors provide the two datasets used for demonstration and evaluation in the paper as ZIP files, each containing:
- `left_rect`: A directory containing rectified RGB images
- `depth`: A directory containing corresponding depth images, in 32-bit floating point format

Unless otherwise specified, all linked datasets and materials are Copyright 2023 Woods Hole Oceanographic Institution.
The datasets are available at: https://drive.google.com/drive/folders/1m64QhlF9vl39eENP9E4tnFF2gc60a1mH?usp=sharing

## BibTeX

```
@INPROCEEDINGS{10160477,
  author={Jamieson, Stewart and How, Jonathan P. and Girdhar, Yogesh},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  title={DeepSeeColor: Realtime Adaptive Color Correction for Autonomous Underwater Vehicles via Deep Learning Methods},
  year={2023},
  volume={},
  number={},
  pages={3095-3101},
  doi={10.1109/ICRA48891.2023.10160477}}
```
