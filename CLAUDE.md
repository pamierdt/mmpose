# CLAUDE.md - AI Assistant Guide for MMPose

## Project Overview

MMPose (v1.3.2) is an OpenMMLab toolbox for pose estimation built on PyTorch. It supports 2D/3D human body, hand, face, whole-body, animal, and fashion landmark pose estimation using both top-down and bottom-up approaches. It is part of the OpenMMLab ecosystem and depends on MMEngine and MMCV.

**License:** Apache 2.0

## Quick Reference

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest tests/

# Run tests with coverage
coverage run --branch --source mmpose -m pytest tests/

# Run a specific test file
pytest tests/test_models/test_backbones/test_hrnet.py

# Lint checks (what CI runs)
flake8 mmpose tests
isort --check-only --diff mmpose tests
yapf -r -d mmpose tests

# Train a model
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py

# Test a model
python tools/test.py <config_file> <checkpoint_file>

# Distributed training
bash tools/dist_train.sh <config_file> <num_gpus>
```

## Repository Structure

```
mmpose/
├── mmpose/                  # Main package source
│   ├── apis/                # High-level inference APIs and inferencers
│   ├── codecs/              # Keypoint encoding/decoding (17+ implementations)
│   ├── configs/             # Runtime Python configs with _base_ inheritance
│   ├── datasets/            # Dataset implementations and transforms
│   ├── engine/              # Training hooks, optimizer wrappers, schedulers
│   ├── evaluation/          # Metrics (AP, AUC, etc.) and evaluators
│   ├── models/              # Neural network components
│   │   ├── backbones/       # 33+ backbone architectures (HRNet, ResNet, Swin, etc.)
│   │   ├── necks/           # Feature pyramid and neck modules
│   │   ├── heads/           # Heatmap, regression, SimCC, transformer heads
│   │   ├── losses/          # Loss functions (heatmap, regression, classification)
│   │   ├── pose_estimators/ # Top-down, bottom-up, pose lifter estimators
│   │   ├── data_preprocessors/
│   │   └── task_modules/
│   ├── registry.py          # All registries (MODELS, DATASETS, TRANSFORMS, etc.)
│   ├── structures/          # PoseDataSample, bbox, keypoint structures
│   ├── utils/               # Utilities, typing aliases, camera tools
│   └── visualization/       # Visualization backends
├── configs/                 # Model configuration files (200+ configs)
│   ├── _base_/datasets/     # Base dataset configurations
│   ├── body_2d_keypoint/    # 2D body pose configs
│   ├── wholebody_2d_keypoint/
│   ├── hand_2d_keypoint/
│   ├── face_2d_keypoint/
│   ├── body_3d_keypoint/
│   ├── animal_2d_keypoint/
│   └── fashion_2d_keypoint/
├── tests/                   # Unit tests (mirrors mmpose/ structure)
│   ├── data/                # Test fixtures and mock data
│   ├── test_apis/
│   ├── test_codecs/
│   ├── test_datasets/
│   ├── test_models/
│   └── test_evaluation/
├── tools/                   # Training, testing, analysis scripts
├── demo/                    # Demo scripts and notebooks
├── docs/                    # Sphinx docs (English + Chinese)
├── projects/                # Community projects (RTMPose, RTMO, etc.)
└── requirements/            # Split dependency files
```

## Architecture & Key Patterns

### Registry System

All components are registered via MMEngine's registry system. New modules must be decorated with the appropriate registry:

```python
from mmpose.registry import MODELS

@MODELS.register_module()
class MyBackbone(BaseBackbone):
    ...
```

Key registries defined in `mmpose/registry.py`:
- `MODELS` - Neural network modules (backbones, heads, losses, estimators)
- `DATASETS` - Dataset classes
- `TRANSFORMS` - Data augmentation/preprocessing transforms
- `KEYPOINT_CODECS` - Keypoint encoding/decoding schemes
- `METRICS` - Evaluation metrics
- `HOOKS` - Training hooks
- `INFERENCERS` - Inference pipeline classes

### Config System

Configs are Python files using inheritance via `_base_`:

```python
_base_ = ['../_base_/datasets/coco.py']

model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='HRNet', ...),
    head=dict(type='HeatmapHead', ...),
)
```

Override config values from CLI with `--cfg-options key=value`.

### Model Architecture

Pose estimators follow a consistent pipeline:
- **Backbone** - Feature extraction (HRNet, ResNet, Swin, etc.)
- **Neck** (optional) - Feature aggregation/FPN
- **Head** - Prediction (heatmap, regression, SimCC, transformer)
- **Codec** - Encode ground truth / decode predictions

The three main estimator types:
1. `TopdownPoseEstimator` - Detect person first, then estimate per-crop
2. `BottomupPoseEstimator` - Detect all keypoints, then group by person
3. `PoseLifter` - Lift 2D predictions to 3D

### Data Flow

`inputs (Tensor)` + `data_samples (List[PoseDataSample])` flow through:
1. `data_preprocessor` - Normalization, padding
2. `backbone` - Feature extraction
3. `neck` (optional) - Feature refinement
4. `head.loss()` (training) or `head.predict()` (inference)
5. Codec decodes head output to keypoint coordinates

### Type Aliases

Common type aliases are in `mmpose/utils/typing.py`:
- `ConfigType` = `dict`
- `OptConfigType` = `Optional[dict]`
- `SampleList` = `List[PoseDataSample]`
- `InstanceList`, `PixelDataList`, `OptMultiConfig`

## Code Style & Conventions

### Formatting Rules

- **Line length:** 79 characters
- **Formatter:** yapf (PEP8 based), configured in `setup.cfg`
- **Import sorting:** isort (line_length=79, multi_line_output=0, known_first_party=mmpose)
- **Docstring wrapping:** 79 characters (docformatter)
- **String quotes:** Double quotes are converted to single quotes (pre-commit hook)
- **Line endings:** LF only
- **Encoding pragma:** Removed (no `# -*- coding: utf-8 -*-`)

### Copyright Header

Every Python file must start with:
```python
# Copyright (c) OpenMMLab. All rights reserved.
```
This is enforced by the `check-copyright` pre-commit hook for `mmpose/`, `tests/`, `demo/`, and `tools/` directories.

### Docstring Style

Google-style docstrings with type annotations:
```python
def method(self, inputs: Tensor, data_samples: SampleList) -> dict:
    """Brief description.

    Args:
        inputs (Tensor): Description with shape (N, C, H, W).
        data_samples (List[:obj:`PoseDataSample`]): Description.

    Returns:
        dict: A dictionary of losses.
    """
```

Docstring coverage must be >= 80% (enforced by `interrogate` in CI).

### Flake8 Exceptions

Config files have relaxed rules (`setup.cfg`):
- `mmpose/configs/*`: F401 (unused import), F403 (wildcard import), F405
- `projects/*/configs/*`: Same exceptions

### Pre-commit Hooks

The full pre-commit pipeline (`.pre-commit-config.yaml`):
1. flake8 (5.0.4)
2. isort (5.11.5)
3. yapf (v0.32.0)
4. trailing-whitespace, check-yaml, end-of-file-fixer, requirements-txt-fixer
5. double-quote-string-fixer (enforces single quotes)
6. check-merge-conflict, mixed-line-ending (LF)
7. docformatter (v1.3.1) - wrap at 79 chars
8. codespell (v2.1.0) - spell checking (skips .ipynb)
9. mdformat (0.7.14) - Markdown formatting with OpenMMlab extensions
10. check-copyright (open-mmlab)

All hooks exclude `tests/data/`.

## Testing

### Framework

- **pytest** with **xdoctest** (auto style) for docstring examples
- Config in `pytest.ini`: `addopts = --xdoctest --xdoctest-style=auto`
- Excluded dirs: `.git`, `build`, `__pycache__`, `data`, `docker`, `docs`, `.eggs`, `.mim`, `tests/legacy`

### Test Organization

Tests mirror the source structure:
- `tests/test_models/` tests `mmpose/models/`
- `tests/test_datasets/` tests `mmpose/datasets/`
- Test data and fixtures live in `tests/data/`

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_codecs/

# With coverage
coverage run --branch --source mmpose -m pytest tests/
coverage report -m

# Single test
pytest tests/test_models/test_backbones/test_hrnet.py -v
```

## Dependencies

### Required (runtime)
- PyTorch >= 1.8
- MMEngine >= 0.6.0, < 1.0.0
- MMCV >= 2.0.0, < 3.0.0
- numpy, scipy, opencv-python, pillow, matplotlib
- xtcocotools >= 1.12
- json_tricks, munkres, chumpy

### Development
- flake8, isort, yapf - Code formatting
- pytest, coverage, xdoctest - Testing
- interrogate - Docstring coverage
- parameterized - Parameterized tests

### Optional
- MMDetection (dev-3.x) - For person detection in top-down pipeline
- albumentations - Advanced augmentation

## CI/CD

### GitHub Actions (`.github/workflows/`)
- **lint.yml** - Code quality on every push/PR
- **pr_stage_test.yml** - Tests on pull requests
- **merge_stage_test.yml** - Full matrix on dev-1.x pushes (Python 3.7-3.9, PyTorch 1.8-2.0, CPU/CUDA/Windows)

### CircleCI (`.circleci/`)
- Dynamic config with path filtering (lint-only vs full test)
- Lint job: flake8, isort, yapf, docstring coverage (80%)
- Build jobs: CPU and CUDA test matrices

## Adding New Components

### New Backbone

1. Create `mmpose/models/backbones/my_backbone.py`
2. Register with `@MODELS.register_module()`
3. Import in `mmpose/models/backbones/__init__.py`
4. Add tests in `tests/test_models/test_backbones/test_my_backbone.py`
5. Create config in `configs/` referencing `type='MyBackbone'`

### New Dataset

1. Create `mmpose/datasets/datasets/<task>/my_dataset.py`
2. Inherit from `BaseCocoStyleDataset` (preferred) or implement custom
3. Register with `@DATASETS.register_module()`
4. Add base config in `configs/_base_/datasets/my_dataset.py`
5. Add dataset metainfo (keypoint definitions, skeleton, etc.)

### New Codec

1. Create `mmpose/codecs/my_codec.py`
2. Register with `@KEYPOINT_CODECS.register_module()`
3. Implement `encode()` and `decode()` methods
4. Add tests in `tests/test_codecs/`

## Common Pitfalls

- Config files use Python syntax, not YAML - use `dict()` or `{}` notation
- The `_base_` inheritance resolves relative to the config file's location
- All model components must be registered before use (import triggers registration)
- `PoseDataSample` is the standard data container - access predictions via `.pred_instances.keypoints`
- Codec `encode()` runs during data loading; `decode()` runs during inference
- Test fixtures in `tests/data/` should not be modified by pre-commit hooks (excluded in config)
- `work_dirs/` and `data/` directories are gitignored - don't commit model weights or datasets
