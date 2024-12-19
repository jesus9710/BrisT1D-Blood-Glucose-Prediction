# BrisT1D Blood Glucose Prediction Competition

This repository contains the code and methodology for a machine learning solution developed to predict blood glucose levels one hour into the future for young adults in the UK with type 1 diabetes. This solution secured 16th place in the 2024 BrisT1D Blood Glucose Prediction Kaggle competition with the following scores:

- **Private Leaderboard RMSE:** 2.4685
- **Public Leaderboard RMSE:** 2.4413

## Dataset

The dataset includes continuous glucose monitor (CGM) readings, insulin pump data, and smartwatch metrics collected from participants. It was released as part of the Kaggle competition and is available [here](https://www.kaggle.com/competitions/brist1d/data).

### Preprocessing

#### 1. Time Window Reduction

The original dataset provided a 6-hour window of past data for predicting blood glucose levels an hour ahead. To optimize for dimensionality and expand the dataset, the time window was reduced to the most recent 4 hours. Dropped columns were then used to rebuild new time-series entries, effectively expanding the dataset.

#### 2. Activity encoding

Activity time-series data was sparse and not suited for one-hot encoding. To address this, an aerobic and anaerobic scoring approach was introduced:

- Activities were scored on a scale of 0 to 3 for both aerobic and anaerobic impact using predefined dictionaries.
- A time penalty factor was calculated for each activity using an inverse sigma-like function to account for temporal effects.
- Two new columns were created: one representing the weighted sum of aerobic components and another for anaerobic components.

These features improved the predictive power of the machine learning models.

#### 3. Handling Missing Values

- Carbohydrate Data: Because the null values represent, on average, 90% of this time serie, the carbohydrate columns were removed.

- Blood Glucose Levels: Missing values were imputed using a multi-step approach:

    - Forward and backward mean filling leveraged strong correlations between time-series elements.
    - For entries with all elements missing, grouping by participant and time was used to impute data.
    - Time-weighted interpolation was applied first to handle grouped missing values, followed by imputation within the grouped dataset.

- Steps, Insulin, Calories, and Heart Rate: Initial attempts used interpolation and forward/backward filling. However, filling missing values with `-1` yielded better results across all models and splits.

#### 4. Feature Engineering

Extensive feature engineering was performed to create new variables, many of which significantly impacted model performance. Key features introduced include:

- **Time Derivative of Blood Glucose:** The approximate derivative of the blood glucose time series at the most recent time step.
- **Relative Blood Glucose Difference:** The difference between the current blood glucose level and the average blood glucose level of the same participant at the same time of day.
- **Maximum Insulin in Recent Steps:** The maximum insulin dosage received in the last 6 time steps.
- **Minutes Since Midnight:** The number of minutes elapsed since 12:00 AM.
- **Time Encoding:** The hour of the day encoded as the cosine of the time (to preserve cyclicity).

Various ratios and interactions between features were also explored. While some yielded inferior results and were discarded, the most impactful transformations were retained and are available in the corresponding feature engineering script.

After all transformations, the time window was further reduced to the most recent **1 hour** of data. This adjustment significantly improved model performance, as the most relevant information for blood glucose prediction lies within the recent time steps.

## Model Architecture

The solution achieving 16th place on the leaderboard employs a stacking regressor ensemble composed of voting regressors built with XGBoost, CatBoost, and LightGBM (LGBM).

## Training

### Data Splitting

To ensure robust training and avoid data leakage, the dataset was split into 9 folds, with grouping by participant. This setup preserved participant-level independence between training and validation data.

### Cross-Validation and Model Training

Each model in the ensemble was trained in a cross-validation context:

- **Voting Regressors**:

    - A total of 9 voting regressors were trained, one for each fold.
    - Each voting regressor consisted of three models of the same architecture.
    - These models were trained on identical datasets but initialized with different random seeds to capture stochastic variations and enhance generalization.
    - The voting regressor classes were designed to be fully compatible with the native training APIs of their respective architectures, allowing GPU acceleration where supported.
- **Note:** LightGBM (LGBM) models were the exception, as they were trained using CPU resources.

### Hyperparameter Optimization

Hyperparameters for all models, except for LightGBM, were optimized using Optuna, a state-of-the-art hyperparameter optimization framework. This approach resulted in significant performance improvements across the ensemble.

## Ensemble

The test dataset posed unique challenges due to higher noise levels and non-overlapping entries (each time series entry was not simply a shifted version of the previous one). To address these issues and enhance robustness, a stacking regressor was employed for the final ensemble.

### Ensemble Methodology

#### 1. Stacking Regressor:

- The stacking regressor used XGBoost as the meta-learner.
- Base models included the previously trained voting regressors, leveraging their diverse predictions for improved accuracy.

#### 2. Implementation:

- The custom stacking regressor class was designed to seamlessly integrate with the original training APIs of the base models.
- Full GPU acceleration was supported for all models where applicable, ensuring efficient training and inference.

This ensemble strategy provided a more reliable solution by aggregating predictions from multiple models, effectively reducing the impact of noise in the test data.

## What did not work:

During development, several alternative approaches were tested but did not yield performance improvements:

- **Interpolation for Blood Glucose Imputation:** Replacing missing blood glucose values with interpolated values instead of the average of forward and backward fills reduced performance.
- **Advanced Imputation for Steps, Insulin, Calories, and Heart Rate:** Using more sophisticated imputation techniques for these features resulted in worse outcomes. Simply imputing missing values with -1 provided better results.
- **Alternative Encoding for Activity Columns:** Attempting one-hot encoding for the original activity columns, as well as various other encoding strategies, consistently led to inferior results.
- **Alternative Time Windows:** Different time windows for the time series data were evaluated. A 1-hour time window consistently delivered the best results.
- **Trend Line Features:** Adding a trend line calculated through linear regression on blood glucose time series data did not improve performance and was ultimately discarded.