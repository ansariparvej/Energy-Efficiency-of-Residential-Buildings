from energy_efficiency.exception import EnergyException
from energy_efficiency.logger import logging
from energy_efficiency.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from energy_efficiency.entity.config_entity import ModelTrainerConfig
import os
import sys
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from energy_efficiency.ml.metric.regression_metric import get_regression_score
from energy_efficiency.ml.model.estimator import EnergyModel
from energy_efficiency.utils.main_utils import load_numpy_array_data, save_object, load_object


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.learning_rate = self.model_trainer_config.learning_rate
            self.n_estimators = self.model_trainer_config.n_estimators
            self.max_depth = self.model_trainer_config.max_depth
            self.col_sample_by_tree = self.model_trainer_config.col_sample_by_tree
        except Exception as e:
            raise EnergyException(e, sys)

    def perform_hyper_parameter_tuning(self): ...

    def train_model(self, x_train, y_train):
        try:
            xgb_clf_obj = XGBRegressor(learning_rate=self.learning_rate, n_estimators=self.n_estimators, max_depth=self.max_depth, colsample_bytree=self.col_sample_by_tree)
            xgb_clf = MultiOutputRegressor(xgb_clf_obj)
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise e
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)


            x_train, y_train, x_test, y_test = (
                train_arr[:, :-2],
                train_arr[:, -2:],
                test_arr[:, :-2],
                test_arr[:, -2:]
            )

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            regression_train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
            
            if regression_train_metric.r2_score <= self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)

            # Over_fitting and Under_fitting
            diff = abs(regression_train_metric.r2_score - regression_test_metric.r2_score)
            
            if diff > self.model_trainer_config.over_fitting_under_fitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            energy_model = EnergyModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=energy_model)

            # model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=regression_train_metric,
            test_metric_artifact=regression_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise EnergyException(e, sys)