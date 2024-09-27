# Microsaccade Selectivity as Discriminative Feature

This repository contains code and resources for [Microsaccade Selectivity as Discriminative Feature for Object Decoding](https://www.biorxiv.org/content/10.1101/2024.04.13.589338v1).
The project explores the effect of stimulus catagories on saccade/microsaccade, particularly microsaccade rate among the different stimulus categories.

## Table of Contents

- [Microsaccade Selectivity as Discriminative Feature](#microsaccade-selectivity-as-discriminative-feature)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Project Structure](#project-structure)
  - [Installation and Setup Instructions](#installation-and-setup-instructions)
  - [Data Availability](#data-availability)
  - [Contributing](#contributing)
  - [Citing This Work](#citing-this-work)
  - [License](#license)

## Abstract

Microsaccades, a form of fixational eye movements, maintain visual stability during stationary observations. Previous studies have provided valuable insights into the relationship between microsaccade characteristics and external stimuli. However, the dynamic nature of microsaccades provides an opportunity to explore the mechanisms of information processing, particularly object decoding. This study examines the modulation of microsaccadic rates by different stimulus categories. Our experimental approach involves an analysis of microsaccade characteristics in monkeys and human subjects engaged in a passive viewing task. The stimulus categories comprised four primary categories: human, animal, natural, and man-made. We identified distinct microsaccade patterns across different stimulus categories, successfully decoding the stimulus category based on the microsaccade rate post-stimulus distribution. Our experiments demonstrate that stimulus categories can be classified with an average accuracy and recall of up to 85%. Our study found that microsaccade rates are independent of pupil size changes. Neural data showed that category classification in the inferior temporal (IT) cortex peaks earlier than microsaccade rates, suggesting a feedback mechanism from the IT cortex that influences eye movements after stimulus discrimination. These results exhibit potential for advancing neurobiological models, developing more effective human-machine interfaces, optimizing visual stimuli in experimental designs, and expanding our understanding of the capability of microsaccades as a feature for object decoding.

**Keywords**: Microsaccade; Eye Movements; Eye Tracking; Object Recognition

## Project Structure

- [Data](./data/): Contains raw samples of human and monkey eye movement data.
- [Scripts](./scripts/): Source code for data preprocessing, statistical analysis, and visualizations. Further information about the scripts is available on [Scripts-Readme](./scripts/README.md).
<!-- - [Notebooks](./notebooks/): Jupyter notebooks for exploration, analysis, and visualizations. -->

## Installation and Setup Instructions

To use the tools and scripts provided in this repository, follow these general setup instructions:

1. **Clone the Repository**: `git clone git@github.com:sa-nouri/ms-selectivity-feature.git`

2. **Environment Setup**:
   - Python: Ensure Python 3.11.7 is installed. Install required libraries using `pip install -r requirements.txt`.

## Data Availability

In [Data](./data/) directory, we have provided sample data from a complete session for both [human](./data/human_sample/) and [monkey](./data/monkey_sample/) subjects. While we recognize the importance of data availability for verification and further research, the entire dataset cannot be made publicly available due to privacy and research lab constraints. However, as mentioned in the Data Availability section of our paper, we are happy to share additional data upon reasonable request from recognized academic or research institutions.

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
