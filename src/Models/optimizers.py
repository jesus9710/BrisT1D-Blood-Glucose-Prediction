import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
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

            num_boost_round = trial.suggest_int('num_boost_round', 50, 500, log=True)

            _df_train = df_train[df_train["fold"] != fold].reset_index(drop=True)
            _df_valid = df_train[df_train["fold"] == fold].reset_index(drop=True)

            dtrain = xgb.DMatrix(_df_train[train_cols], label=_df_train["bg+1:00"])
            dval = xgb.DMatrix(_df_valid[train_cols], label = _df_valid["bg+1:00"])

            gbm = xgb.train(param, dtrain, num_boost_round, evals=[(dval,'validation')], early_stopping_rounds=50, verbose_eval=False)
            
            preds = gbm.predict(dval)
            score = root_mean_squared_error(_df_valid["bg+1:00"], preds)

            scores.append(score)

        return np.mean(scores)
    
    return objective

def get_optimizer_for_CatBoost(df_train, n_splits, train_cols):

    def objective(trial):
        param = {
            "task_type": "GPU",
            "loss_function": "RMSE",
            "depth": trial.suggest_int("depth", 3,7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg",1.0,10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature",1.0,10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "verbose":False
        }
        scores = []
        for fold in range(n_splits):

            _df_train = df_train[df_train["fold"] != fold].reset_index(drop=True)
            _df_valid = df_train[df_train["fold"] == fold].reset_index(drop=True)

            dtrain = Pool(_df_train[train_cols], label=_df_train["bg+1:00"])
            model = CatBoostRegressor(**param)
            dval = Pool(_df_valid[train_cols], _df_valid["bg+1:00"])

            cbm = model.fit(dtrain,
                            eval_set=dval,
                            early_stopping_rounds=30)

            preds = cbm.predict(dval)
            score = root_mean_squared_error(_df_valid["bg+1:00"], preds)

            scores.append(score)

        return np.mean(scores)
    
    return objective

def get_optimizer_for_LGBM(df_train, n_splits, train_cols):

    def objective(trial):
        param = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves" : trial.suggest_int("num_leaves",20,300),
            "max_depth" : trial.suggest_int("max_depth",3,15),
            "n_estimators" : trial.suggest_int("n_estimators",100,10000),
            "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf",10,100),
            "lambda_l1" : trial.suggest_float("lambda_l1",0.0,10.0),
            "lambda_l2" : trial.suggest_float("lambda_l2",0.0,10.0),
            "feature_fraction" : trial.suggest_float("feature_fraction",0.3,1.0),
            "bagging_fraction" : trial.suggest_float("bagging_fraction",0.5,1.0),
            "bagging_freq" : trial.suggest_int("bagging_freq",1,5)
        }

        scores = []
        for fold in range(n_splits):

            _df_train = df_train[df_train["fold"] != fold].reset_index(drop=True)
            _df_valid = df_train[df_train["fold"] == fold].reset_index(drop=True)

            dtrain = Pool(_df_train[train_cols], label=_df_train["bg+1:00"])
            model = CatBoostRegressor(**param)
            dval = Pool(_df_valid[train_cols], _df_valid["bg+1:00"])

            cbm = model.fit(dtrain,
                            eval_set=dval,
                            early_stopping_rounds=30)

            preds = cbm.predict(dval)
            score = root_mean_squared_error(_df_valid["bg+1:00"], preds)

            scores.append(score)

        return np.mean(scores)
    
    return objective