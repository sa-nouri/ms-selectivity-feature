# Microsaccade Selectivity as Discriminative Feature for Object Decoding

This repository contains the code for analyzing microsaccade selectivity as a discriminative feature for object decoding in eye tracking data.

## Features

- Microsaccade detection and analysis
- Blink detection and removal
- Glitch detection and correction
- Data preprocessing and postprocessing
- Comprehensive test suite
- Logging support

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ms-selectivity-feature.git
cd ms-selectivity-feature
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv msenv
source msenv/bin/activate  # On Windows: msenv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

For development, install with additional dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from src import detect_microsaccades, detect_blinks, detect_glitches
import numpy as np

# Load your eye tracking data
timestamps = np.load('data/timestamps.npy')
x_positions = np.load('data/x_positions.npy')
y_positions = np.load('data/y_positions.npy')

# Detect microsaccades
microsaccades = detect_microsaccades(
    x_positions,
    y_positions,
    timestamps,
    velocity_threshold=6.0,
    min_duration=3,
    max_duration=20
)

# Detect blinks
blinks = detect_blinks(
    x_positions,
    y_positions,
    timestamps,
    min_duration=50,
    min_amplitude=2.0
)

# Detect glitches
glitches = detect_glitches(
    x_positions,
    y_positions,
    timestamps,
    threshold=5.0
)
```

### Advanced Usage

For more advanced usage examples, please refer to the [examples](examples/) directory.

## Development

### Setting Up Development Environment

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
pytest
```

For coverage report:
```bash
pytest --cov=src tests/
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks:
```bash
black src tests
isort src tests
flake8 src tests
mypy src
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper,
  title={Microsaccade Selectivity as Discriminative Feature for Object Decoding},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Acknowledgments

- List any acknowledgments here
- Include references to related work
- Credit any collaborators or institutions
