# DataMining-Project: Chess Evaluation Prediction

## Description

This project aims to predict chess position evaluations (in centipawns) directly from their Forsyth–Edwards Notation (FEN) representation using data mining and machine learning techniques. It explores various regression models to estimate the evaluation score typically provided by strong chess engines like Stockfish.

## Data

The models are trained and evaluated using chess datasets containing FEN strings and corresponding engine evaluations. The primary datasets used are:

* `chessData.csv`  - used for training the models.
* `random_evals.csv` - used for testing or validation.

These datasets are expected to be available, potentially within a Kaggle environment under `/kaggle/input/chess-evaluations/` or `/kaggle/input/chess-dataset/`.

## Feature Extraction

FEN strings are parsed to extract a variety of features for the models, including:

* **Piece Counts**: Number of each piece type (Pawn, Knight, Bishop, Rook, Queen, King) for both white and black.
* **Material Balance**: Calculated based on standard piece values.
* **Active Player**: Whose turn it is ('w' or 'b').
* **Castling Rights**: Availability of kingside and queenside castling for both players.
* **King Castled Status**: Whether each king has already castled.
* **Pawn Structure**: Metrics related to pawn advancement.
* **Piece Positions**: Queen centrality, center control by pieces.
* **Engine Features (XGBoost specific)**: Stockfish depth=1 evaluation and piece-square table (PST) scores are also used in the `xgb-model.ipynb` notebook.

Parallel processing (`joblib`, `multiprocessing`) is employed for efficient feature extraction from large datasets.

## Models Implemented

This project implements and evaluates several machine learning models:

1.  **Random Forest**: Using `sklearn.ensemble.RandomForestRegressor`.
2.  **XGBoost**: Using the `xgboost` library, potentially with GPU acceleration and hyperparameter tuning via Optuna.
3.  **LightGBM**: Using the `lightgbm` library.

## Preprocessing

* Raw evaluation strings (e.g., '+56', '#-3') are cleaned and converted to numerical centipawn values.
* Evaluations are typically clipped (e.g., to +/- 1000 cp) and scaled (e.g., to pawn units by dividing by 100).
* Features are standardized using `sklearn.preprocessing.StandardScaler`.
* Preprocessing steps like feature extraction and scaling results might be cached (e.g., to `/kaggle/working/chess_cache/` or `/kaggle/working/`).

## Setup and Dependencies

The project is designed to run in a Python 3 environment, likely on Kaggle[cite: 4]. Key dependencies include:

* `pandas`, `numpy`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `python-chess`
* `stockfish` (Python wrapper and engine binary)
* `joblib`
* `matplotlib`
* `optuna` (for XGBoost tuning)

The notebooks often include installation commands (`!pip install ...`, `!apt-get install ...`) for these dependencies.

## How to Run

Refer to the specific Jupyter Notebooks (`*.ipynb`) for detailed code execution. The `How_to_run.pdf` provides specific instructions for running the `xgb-model.ipynb` on Kaggle:

1.  **Environment**: Use a Kaggle Notebook with a GPU accelerator (e.g., GPU P100 recommended for XGBoost).
2.  **Data**: Add the "Chess Evaluations" dataset (e.g., from user Ronak Badhe on Kaggle). Ensure the CSV files (`chessData.csv`, `random_evals.csv`) are accessible at the paths expected by the notebook (e.g., `/kaggle/input/chess-evaluations/` or `/kaggle/input/chess-dataset/`)[cite: 1, 2, 3, 4].
3.  **Notebook**: Upload the desired notebook file (e.g., `xgb-model.ipynb`, `lightgbm-model.ipynb`, `random-forest.ipynb`).
4.  **Execution**: Run all cells using "Run → Run All". Note that feature extraction can take significant time (potentially hours). Training time varies by model and hardware (e.g., 10-20 minutes for GPU XGBoost).

## Outputs

Execution typically generates:

* Extracted features (e.g., `.parquet`, `.pkl` files).
* Trained models (e.g., `.joblib` files).
* Scalers (e.g., `.joblib` files).
* Evaluation results and plots.

These outputs are usually saved in the `/kaggle/working/` directory or a specified sub-directory like `/kaggle/working/chess_cache/`.

## Evaluation

Model performance is assessed using:

* **Root Mean Squared Error (RMSE)**: Measured in centipawns or pawn units.
* **R² Score**: Coefficient of determination.
* **Mean Absolute Error (MAE)**: Used in benchmarking.
* **Accuracy**: Percentage of predictions within a certain threshold (e.g., ±1 pawn).

Benchmarking may compare model predictions against deeper Stockfish evaluations (e.g., depth 12) on both full-range and mid-game positions.
