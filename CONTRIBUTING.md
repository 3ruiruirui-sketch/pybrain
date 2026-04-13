# Contributing to PY-BRAIN

Thank you for your interest in contributing to PY-BRAIN. This document
provides guidelines and instructions for contributing.

## Quick Links

- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
- [Validation Documentation](./docs/technical/VALIDATION.md)
- [Model Card](./docs/technical/MODEL_CARD.md)

---

## Ways to Contribute

You can contribute to PY-BRAIN in several ways:

- **Report bugs** — Submit issues for unexpected behavior, crashes, or incorrect results
- **Suggest features** — Open issues to propose new capabilities or model integrations
- **Improve documentation** — Fix ambiguities, add examples, or translate
- **Submit code** — Pull requests for bug fixes, optimizations, or new features
- **Validate results** — Run the pipeline on new BraTS cases and report metrics

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Internet access (for downloading model weights on first run)
- macOS, Linux, or Windows with WSL2

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/3ruiruirui-sketch/pybrain.git
cd pybrain

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; import monai; print('OK')"
```

### Environment Variables

The pipeline uses `PYBRAIN_SESSION` for deterministic runs:

```bash
export PYBRAIN_SESSION="my_experiment_001"
export PYBRAIN_SEED=42  # Random seed for reproducibility
```

---

## Coding Standards

### Python Style

- Follow **PEP 8** with 120-character line length limit
- Use `ruff` for linting: `ruff check pybrain/ scripts/`
- Use `pyright` for type checking: `pyright pybrain/`

### Documentation

- All public functions and classes should have docstrings (NumPy style)
- Complex scientific logic should include inline comments explaining the rationale
- Update relevant documentation (README.md, docs/) when changing behavior

### Testing

- Add unit tests for new functions in `tests/`
- Run existing tests: `python -m pytest tests/ -v`
- **Important**: Do not commit tests that require model weight downloads or
  large datasets — mark them with `@pytest.mark.slow` and skip in CI

### Git Commit Messages

Use clear, descriptive commit messages:

```
feat: add Platt calibration for per-subregion probability output
fix: correct ET label from 3 to 4 in BraTS convention (issue #XX)
docs: add SwinUNETR validation protocol to VALIDATION.md
refactor: extract ensemble weights to config/defaults.yaml
```

---

## Pipeline Structure

The pipeline consists of numbered stages in `scripts/`:

| Stage | Script | Purpose |
|-------|--------|---------|
| 0 | `0_preflight_check.py` | Environment and data validation |
| 1 | `1_dicom_to_nifti.py` | DICOM → NIfTI conversion |
| 2 | `2_ct_integration.py` | CT registration and calcification masks |
| **3** | **`3_brain_tumor_analysis.py`** | **AI segmentation (main entry point)** |
| 5 | `5_validate_segmentation.py` | Validation metrics |
| 6 | `6_tumour_location.py` | Anatomical location analysis |
| 7 | `7_tumour_morphology.py` | Morphological metrics |
| 8 | `8_radiomics_analysis.py` | Radiomics + ML classification |
| 9 | `9_generate_report.py` | PDF report generation |

Run the full pipeline:
```bash
python scripts/3_brain_tumor_analysis.py
```

---

## Scientific Contribution Guidelines

When contributing scientific functionality:

1. **Reproducibility**: Ensure all random seeds are documented and settable
2. **Validation**: Add metrics to `docs/technical/VALIDATION.md` with clear targets
3. **Configuration**: All hyperparameters must be in `pybrain/config/defaults.yaml`
4. **No hardcoded values**: Thresholds, weights, and parameters belong in config
5. **Ethical considerations**: Clearly document limitations and clinical disclaimers

---

## Pull Request Process

### Before Submitting

- [ ] Run `ruff check .` and fix all warnings
- [ ] Run `pyright pybrain/` and resolve all type errors
- [ ] Add or update tests for changed functionality
- [ ] Update documentation if behavior changed
- [ ] Ensure `PYBRAIN_SEED=42` produces deterministic results

### Pull Request Description

Include:
- Summary of changes
- Link to related issue (e.g., "Closes #12")
- Validation results on at least one BraTS case
- Any new dependencies added

### Review Process

PRs are reviewed by maintainers. Reviews focus on:
- Scientific correctness and reproducibility
- Code quality and type safety
- Adequate test coverage
- Documentation accuracy

---

## Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, PyTorch version, MONAI version
2. **Command**: Exact command that caused the bug
3. **Error message**: Full traceback
4. **Data**: BraTS case ID if applicable (e.g., BraTS2021_00000)
5. **Expected vs actual behavior**

**Do not include patient data** in bug reports. Use synthetic or
publicly available BraTS data for reproduction.

---

## Feature Requests

For significant features (new models, new segmentation paradigms), please
open an issue first to discuss design before investing time in implementation.

---

## License

By contributing, you agree that your contributions will be licensed under the
MIT License. See [LICENSE](./LICENSE) for details.
