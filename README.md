# Microsaccade Selectivity as Discriminative Feature

This repository contains code and resources for [Microsaccade Selectivity as Discriminative Feature for Object Decoding](https://www.biorxiv.org/content/10.1101/2024.04.13.589338v1).
The project explores the effect of stimulus catagories on saccade/microsaccade, particularly microsaccade rate among the different stimulus categories.

## Table of Contents

- [Microsaccade Selectivity as Discriminative Feature](#microsaccade-selectivity-as-discriminative-feature)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Project Structure](#project-structure)
    - [**Notebooks**](#notebooks)
    - [**SRC**](#src)
  - [Installation and Setup Instructions](#installation-and-setup-instructions)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Citing This Work](#citing-this-work)
  - [License](#license)

## Abstract

Microsaccades, a form of fixational eye movements, maintain visual stability during stationary observations. Previous studies have provided valuable insights into the relationship between microsaccade characteristics and external stimuli. However, the dynamic nature of microsaccades provides an opportunity to explore the mechanisms of information processing, particularly object decoding. This study examines the modulation of microsaccadic rates by different stimulus categories. Our experimental approach involves an analysis of microsaccade characteristics in monkeys and human subjects engaged in a passive viewing task. The stimulus categories comprised four primary categories: human, animal, natural, and man-made. We identified distinct microsaccade patterns across different stimulus categories, successfully decoding the stimulus category based on the microsaccade rate post-stimulus distribution. Our experiments demonstrate that stimulus categories can be classified with an average accuracy and recall of up to 85%. These results exhibit potential for advancing neurobiological models, developing more effective human-machine interfaces, optimizing visual stimuli in experimental designs, and expanding our understanding of the capability of microsaccades as a feature for object decoding.

**Keywords**: Microsaccade; Eye Movements; Eye Tracking; Object Recognition

## Project Structure

- [Notebooks](./notebooks/): Jupyter notebooks for exploration and analysis.
- [SRC](./src/): Source code for data preprocessing, model training, statistical analysis, and graphing

### **Notebooks**

- [`analyzing_pacs`](./notebooks/analyzing_pacs/): Explore and visualize the generated PAC data.
- [`analyzing_saliency_maps`](./notebooks/analyzing_saliency_maps/): Explore and visualize the generated saliency maps data.
- [`grad_camp`](./notebooks/grad_cam/): Apply an example of Grad-CAM as an explainable AI method.

### **SRC**

- [PACNET](./src/PACNET/): Implements the generation of raw PAC data and images, utilizes deep transfer learning based on VGG16 for estimating motor vigor from raw PAC data, and includes Grad-CAM implementation for feature visualization. Additionally, it performs correlation analysis for representational similarity among features and raw PACs.
- [Headplots](./src/Headplots/): Visualizes headplots using MATLAB
- [Lasso](./src/Lasso/): Employs Lasso regression in MATLAB to estimate motor vigor from selected features and PAC data.
- [Statistical Analysis](./src/Statistical%20Analysis/): Conducts statistical analysis in R, examining the relationships and effects within PAC data and saliency maps under various conditions.

## Installation and Setup Instructions

To use the tools and scripts provided in this repository, follow these general setup instructions:

1. **Clone the Repository**: `git clone git@github.com:sa-nouri/ms-selectivity-feature.git`

2. **Environment Setup**:
   - Python: Ensure Python 3.11.7 is installed. Install required libraries using `pip install -r requirements.txt`.
   - MATLAB: Ensure MATLAB is installed with the necessary toolboxes for data analysis.

## Usage

For detailed usage instructions, including how to run each component of the project, please refer to the [SRC README.md](./src/README.md) file.

## Contributing

If you want to contribute to this project, please check our [contributing guidelines](./CONTRIBUTING.md).

## Citing This Work

If you use the "ms-selectivity-features" project or the research it is based on for your own work, please consider citing the following paper:

```bibtex
@article{nouri2024microsaccade,
  title={Microsaccade Selectivity as Discriminative Feature for Object Decoding},
  author={Nouri, Salar and Soltani Tehrani, Amirali and Faridani, Niloufar and Toosi, Ramin and Dehaqani, Mohammad-Reza A},
  journal={bioRxiv},
  pages={2024--04},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the [MIT License](./LICENSE.md).