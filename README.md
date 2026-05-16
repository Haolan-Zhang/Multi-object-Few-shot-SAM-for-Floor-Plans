# Multi-object Few-shot SAM for Floor Plans

Reference implementation for the paper
**"Few-shot learning with large foundation models for automated segmentation
and accessibility analysis in architectural floor plans."**

## Status

The full implementation will be released after my thesis defense
(expected end of 2026). In the meantime, this repository provides the data
preprocessing pipeline and a small proof of concept. Combined with the
pseudo-code in the paper, the remaining components should be straightforward
to reproduce.

Roadmap:

- [x] Data preprocessing
- [ ] Proof of concept
- [ ] Full implementation

## Installation

```bash
pip install opencv-python numpy cairosvg pillow tqdm
```

The preprocessing script reuses the `floortrans` package that ships with
[CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k), which is already
included in this repository.

## Data Preprocessing

`extract_cubicasa_data.py` renders CubiCasa floor plans, extracts per-room
masks, and saves a single selected room mask for each sample.

### 1. Download the dataset

Download the CubiCasa5K floor plans from
[Zenodo](https://zenodo.org/record/2613548), unzip the archive, and place
the contents under the `data/` directory:

```
data/
└── cubicasa5k/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── ...
```

### 2. Run the extractor

```bash
python extract_cubicasa_data.py \
  --txt test.txt \
  --svg-dir svg_images_new \
  --npy-dir mask_npys_new \
  --selected-mask-dir train_masks_new
```

### 3. Outputs

| Flag                   | Contents                                                          |
| ---------------------- | ----------------------------------------------------------------- |
| `--svg-dir`            | Rendered `model.svg` floor plans saved as `.jpg`                  |
| `--npy-dir`            | All room masks per sample as `.npy` with shape `(num_rooms, H, W)`|
| `--selected-mask-dir`  | One selected room mask per sample saved as `.png`                 |

### Useful options

| Flag                | Default              | Description                                       |
| ------------------- | -------------------- | ------------------------------------------------- |
| `--data-path`       | `data/cubicasa5k/`   | Root directory of the CubiCasa5K dataset          |
| `--txt`             | `test.txt`           | Split file under `--data-path`                    |
| `--start-index`     | `0`                  | First dataset index to process                    |
| `--max-samples`     | `None`               | Process at most this many samples                 |
| `--selected-rank`   | `2`                  | Save the Nth largest room as the selected mask    |
| `--original-size`   | off                  | Use original-size labels instead of scaled labels |
| `--overwrite`       | off                  | Overwrite existing files in the output directories|

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2025few,
  title   = {Few-shot learning with large foundation models for automated segmentation and accessibility analysis in architectural floor plans},
  author  = {Zhang, Haolan and Zhang, Ruichuan},
  journal = {Journal of Infrastructure Intelligence and Resilience},
  volume  = {4},
  number  = {2},
  pages   = {100137},
  year    = {2025},
  publisher = {Elsevier}
}
```
