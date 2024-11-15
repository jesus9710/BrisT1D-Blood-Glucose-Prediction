import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

def get_optimizer_for_XGBoost(df_train, n_splits, train_cols):

    def objective(trial):
        param = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001,20.0),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "gamma": trial.suggest_int("gamma", 0, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "max_leaves": trial.suggest_int("max_leaves", 2, 256),
            "device": "gpu"
        }
        scores = []
        for fold in range(n_splits):

            _df_train = df_train[df_train["fold"] != fold].reset_index(drop=True)
            _df_valid = df_train[df_train["fold"] == fold].reset_index(drop=True)

            dtrain = xgb.DMatrix(_df_train[train_cols], label=_df_train["bg+1:00"])
            gbm = xgb.train(param, dtrain)

            dval = xgb.DMatrix(_df_valid[train_cols])
            preds = gbm.predict(dval)
            score = root_mean_squared_error(_df_valid["bg+1:00"], preds)

            scores.append(score)

        return np.mean(scores)
    
    return objective