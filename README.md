# Rocket League Skill-Shot Classification

A machine-learning notebook that classifies Rocket League gameplay sequences into six advanced mechanics or noise. The project parses variable-length telemetry samples, engineers fixed-length statistical features, and trains a multiclass linear classifier with PyTorch.

## Project overview

The notebook provides an end-to-end classification workflow:

1. Parse the bundled `.data` file into labeled, variable-length sequences.
2. Engineer a fixed-length representation using summary statistics, controller-input counts, and sample duration.
3. Explore class balance and feature correlations.
4. Create a stratified train/test split and standardize the features.
5. Train a seven-class linear model with cross-entropy loss and Adam.
6. Evaluate the model with accuracy, precision, recall, F1 score, and a confusion matrix.

The dataset contains 298 samples, 18 raw telemetry fields, and the following labels:

| Raw label | Class |
| ---: | --- |
| -1 | Noise |
| 1 | Ceiling shot |
| 2 | Power shot |
| 3 | Waving dash |
| 5 | Air dribble |
| 6 | Front flick |
| 7 | Musty flick |

Feature engineering produces 84 model inputs by calculating the mean, standard deviation, minimum, and maximum of each telemetry field; summing binary controller inputs; and recording sequence duration.

## Repository structure

```text
.
├── data/
│   └── rocket_league_skillshots.data
├── rocket_league_classification.ipynb
├── requirements.txt
└── README.md
```

## Getting started

Python 3.13 is recommended because it matches the notebook's current kernel metadata.

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repository-url>
   cd CS222-final_project-EthanBennett
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Start JupyterLab:

   ```bash
   jupyter lab rocket_league_classification.ipynb
   ```

5. Run the notebook cells from top to bottom.

Run JupyterLab from the repository root so the relative dataset path resolves correctly.

## Example results

The output saved in the notebook records:

- 89.69% training accuracy after 10 epochs
- 85.33% test accuracy on 75 held-out samples
- 0.852 macro-average F1 score

These values are a reference rather than a guaranteed benchmark. The train/test split and model initialization are not seeded, so results can vary between runs.

## Model details

The classifier is a single fully connected layer with seven output logits. Training uses cross-entropy loss, the Adam optimizer, a learning rate of `0.01`, a batch size of `32`, and `10` epochs. Input features are standardized using statistics learned only from the training split.

## Notes

- The dataset is included in the repository; no separate download is required.
- The data file must remain at `data/rocket_league_skillshots.data` unless the notebook's `data_path` is updated.
- No license is currently included. All rights remain with the repository owner unless a license is added.
