import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool, CatBoostRegressor
import lightgbm as lgb 
from pathlib import Path

class Base_VotingRegressor:

    def __init__(self, n_models, params = None):

        self.n_models = n_models
        self.params = params
        self.trained_models = []
        self.predictions = []
        self.averaged_predictions = None
        self.feature_importances = None
    
    def get_params(self, params):

        self.params = params

        return self.params
    
    def fit(self, dtrain, fit_params = {}):

        self.trained_models = []
        params = self.params.copy()

        for idx in range(self.n_models):

            self.trained_models.append(self._fit(dtrain, idx, params, fit_params))

        return self.trained_models
    
    def predict(self, dtest):

        self.predictions = []

        for model in self.trained_models:

            self.predictions.append(model.predict(dtest))

        self.averaged_predictions = np.mean(self.predictions, axis=0)

        return self.averaged_predictions
    
    def get_feature_importances(self):

        self.feature_importances = pd.Series()

        for model in self.trained_models:

            fi = self._get_feature_importances(model)
            fi_serie = pd.Series(fi).sort_values(ascending=False)
            
            self.feature_importances = self.feature_importances.add(fi_serie, fill_value=0)
        
        self.feature_importances = (self.feature_importances / len(self.trained_models)).sort_values(ascending=False)

        return self.feature_importances
    
    def _fit(self, dtrain, idx, params, fit_params):
        pass

    def _get_feature_importances(self, model):
        pass


class XGB_VotingRegressor(Base_VotingRegressor):

    def __init__(self, n_models, params = None):
        super().__init__(n_models, params)
    
    def _fit(self, dtrain, idx, params, fit_params):

        params.update({'seed': 42 + idx})

        return xgb.train(params, dtrain, **fit_params)
    
    def _get_feature_importances(self, model):

        return model.get_score(importance_type='weight')
    
    def save_model(self, file_name):

        for idx, model in enumerate(self.trained_models):

            model.save_model(file_name / (file_name.name + '_' + str(idx) + '.bin'))

    def get_feature_names(self):

        return self.trained_models[0].feature_names

    @classmethod
    def load(cls, path):

        files = list(Path.glob(path,'*.bin'))

        model = XGB_VotingRegressor(n_models=len(files))

        for f in files:
            
            booster = xgb.Booster()
            booster.load_model(f)
            model.trained_models.append(booster)
        
        return model

class CatBst_VotingRegressor(Base_VotingRegressor):

    def __init__(self, n_models, params=None):
        super().__init__(n_models, params)

    def _fit(self, dtrain, idx, params, fit_params):

        params.update({'random_seed': 42 + idx})
        model = CatBoostRegressor(**params)

        return model.fit(dtrain, **fit_params)
    
    def _get_feature_importances(self, model):

        values = model.get_feature_importance()
        names = model.feature_names_

        fi_dict = {key:value for key, value in zip(names, values)}

        return fi_dict
    
    def save_model(self, file_name):

        for idx, model in enumerate(self.trained_models):

            model.save_model(file_name / (file_name.name + '_' + str(idx) + '.cbm'))

    def get_feature_names(self):

        return self.trained_models[0].feature_names_

    @classmethod
    def load(cls, path):

        files = list(Path.glob(path,'*.cbm'))

        model = CatBst_VotingRegressor(n_models=len(files))

        for f in files:
            
            booster = CatBoostRegressor()
            booster.load_model(f)
            model.trained_models.append(booster)
        
        return model


class LGBM_VotingRegressor(Base_VotingRegressor):

    def __init__(self, n_models, params = None):
        super().__init__(n_models, params)
        
    def _fit(self, dtrain, idx, params, fit_params):

        params.update({'seed': 42 + idx})

        return lgb.train(params, dtrain, **fit_params)
    
    def _get_feature_importances(self, model):

        values = model.feature_importance()
        names = model.feature_name()

        fi_dict = {key:value for key, value in zip(names, values)}

        return fi_dict
    
    def save_model(self, file_name):

        for idx, model in enumerate(self.trained_models):

            model.save_model(file_name / (file_name.name + '_' + str(idx) + '.bin'))

    def get_feature_names(self):

        return self.trained_models[0].feature_name()

    @classmethod
    def load(cls, path):

        files = list(Path.glob(path,'*.bin'))

        model = LGBM_VotingRegressor(n_models=len(files))

        for f in files:
            
            booster = lgb.Booster(model_file=f)
            model.trained_models.append(booster)
        
        return model

class Base_Stacking_Regressor:

    def __init__(self, *estimators):

        self.estimators = estimators
        self.trained_meta_model = None

    def fit_meta(self, data, label, fit_params = {}):

        predictions = self._base_predict(data, label)
        self.trained_meta_model = self._fit_meta(predictions, label, fit_params)
        
        return self.trained_meta_model
    
    def predict(self, dtest):

        predictions = self._base_predict(dtest)

        return predictions
    
    def _base_predict(self, data, label = None):

        predictions = []

        for estimator in self.estimators:

            adapted_data = self._adapt_dataset(estimator, data, label)
            predictions.append(estimator.predict(adapted_data))

        predictions_array = np.array(predictions).transpose()

        return predictions_array
    
    def _adapt_dataset(self, estimator, data, label = None):

        if isinstance(estimator, XGB_VotingRegressor):
            data = xgb.DMatrix(data, label)
        
        elif isinstance(estimator, CatBoostRegressor):
            data = Pool(data, label)

        elif isinstance(estimator, LGBM_VotingRegressor):
            # data = lgb.Dataset(data, label)
            pass

        return data

class XGB_Stacking_Regressor(Base_Stacking_Regressor):

    def __init__(self, meta_learner_params, *estimators):

        super().__init__(*estimators)

        self.params = meta_learner_params

    def predict(self, dtest):

        base_predictions = super().predict(dtest)
        base_predictions = xgb.DMatrix(base_predictions)

        return self.trained_meta_model.predict(base_predictions)

    def _fit_meta(self, Xdata, label, fit_params):

        d_train = xgb.DMatrix(Xdata, label)

        return xgb.train(self.params, d_train, **fit_params)

