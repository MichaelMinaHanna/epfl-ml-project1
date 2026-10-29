# EPFL Machine Learning — Project 1

**Binary Classification of Cardiovascular Disease Risk**

This repository contains our solution for Project 1 of the EPFL Machine Learning course. The objective is to predict the presence of **myocardial infarction or coronary heart disease (MICHD)** from survey data using binary classification methods. The project covers data preprocessing, model implementation, cross-validation, model selection, and prediction generation.

## Team

- Michael Hanna
- Tarek Hamad
- Rex Baseden

## Approach

The main pipeline in `run.py`:

1. **Load data:** Read the provided training and test CSV files. A local NumPy cache (`dataset/cache.npz`) speeds up subsequent runs.
2. **Preprocess features:** One-hot encode columns with up to 10 distinct non-missing values, and add missing-value indicators for continuous features where needed.
3. **Handle missing values and scaling:** Within each cross-validation fold, estimate imputation values and standardization statistics from the training partition, then apply them to training and validation data. Add a bias column.
4. **Compare models:** Evaluate ridge regression, logistic regression, and regularized logistic regression across hyperparameter choices.
5. **Select a threshold:** Choose a classification threshold using out-of-fold predictions to maximize the F1 score over the thresholds considered.
6. **Generate predictions:** Retrain the selected model on the full training set, predict the test set, and export a submission CSV.

The repository also contains implementations of least-squares regression and gradient-based optimization methods for the course project.

## Repository structure

```text
epfl-ml-project1/
├── implementations.py           # Machine learning algorithms
├── helpers.py                   # Provided CSV loading and submission utilities
├── run.py                       # Model comparison and final prediction pipeline
├── explore.ipynb                # Data exploration and model experiments
├── plot.ipynb                   # Result visualization
├── environment.yml              # Original course Conda environment
├── environment-macos-arm64.yml  # Conda environment for Apple Silicon Macs
├── figures/                     # Project figures
├── report/                      # LaTeX project report
└── README.md
```

The `dataset/` folder is created locally by the user and is not committed to the repository. The `submissions/` folder is created automatically when the pipeline runs.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MichaelMinaHanna/epfl-ml-project1.git
cd epfl-ml-project1
```

### 2. Install dependencies

We recommend Conda to reproduce the supplied environment:

```bash
conda env create -f environment.yml
conda activate project1-grading
```

For Apple Silicon Macs, use the dedicated environment file instead:

```bash
conda env create -f environment-macos-arm64.yml
```

Then activate the environment using the name specified in that YAML file.

### 3. Add the dataset

Download the course-provided data and place the three CSV files in a `dataset/` directory **at the repository root**:

```text
epfl-ml-project1/
└── dataset/
    ├── x_train.csv
    ├── y_train.csv
    └── x_test.csv
```

Keep the original filenames. The dataset is not included in the Git repository.

## Run the project

From the repository root, execute:

```bash
python run.py
```

The script evaluates candidate models using **5-fold cross-validation**, reports their F1 score and accuracy, selects the model with the highest cross-validation F1 score, and generates:

```text
submissions/submission.csv
```

The submission contains `Id` and `Prediction` columns, with predictions encoded as `-1` or `1`.

**Note:** The first run may take longer because the CSV files are loaded and cached. Later runs can reuse the local cache.

## Models and evaluation

The project includes implementations of:

- Least-squares regression and mean squared error optimization using gradient descent or stochastic gradient descent
- Ridge regression
- Logistic regression
- Regularized logistic regression

The main pipeline compares ridge regression, logistic regression, and regularized logistic regression. Evaluation uses **F1 score** as the model-selection criterion, with accuracy also reported. The data split is reproducible through a fixed random seed (`42`).

The parameters and preprocessing options can be adjusted directly in `run.py`, including:

| Setting | Default | Purpose |
| --- | --- | --- |
| `K_FOLDS` | `5` | Number of cross-validation folds |
| `SEED` | `42` | Random seed for fold generation |
| `ONE_HOT` | `True` | Enable categorical feature encoding |
| `IMPUTE` | `"median"` | Missing-value replacement (`"median"` or `"zero"`) |
| `STANDARDIZE` | `True` | Enable feature standardization |
| `MAX_ITERS` | `1000` | Gradient descent iterations |
| `GAMMA` | `0.1` | Learning rate |

The preprocessing switches also support the ablation experiments discussed in the project report.

## Report and notebooks

- **`report/`** contains the LaTeX source of the written project report.
- **`explore.ipynb`** contains exploratory analysis and model experiments.
- **`plot.ipynb`** contains result visualizations.
- **`figures/`** stores figures associated with the project.

## Acknowledgments

Developed as part of the **Machine Learning course at École Polytechnique Fédérale de Lausanne (EPFL)**. The project uses course-provided data and helper utilities.
