import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from energy_efficiency.constant.training_pipeline import TARGET_COLUMN
from energy_efficiency.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from energy_efficiency.entity.config_entity import DataTransformationConfig
from energy_efficiency.exception import EnergyException
from energy_efficiency.logger import logging
from energy_efficiency.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise EnergyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EnergyException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_impute = SimpleImputer(strategy="constant", fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ("Impute", simple_impute),  # replace missing values with zero
                    ("RobustScaler", robust_scaler)  # keep every feature in same range and handle outlier
                    ])
            return preprocessor

        except Exception as e:
            raise EnergyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # training dataframe

            input_feature_train_df = train_df.drop(columns=TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # testing dataframe

            input_feature_test_df = test_df.drop(columns=TARGET_COLUMN, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Preprocessing object
            preprocessor_object = DataTransformation.get_data_transformer_object()
            transformed_input_train_feature = preprocessor_object.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            # preprocessor = self.get_data_transformer_object()
            # preprocessor_object = preprocessor.fit(input_feature_train_df)
            # transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            # transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            input_feature_train_final, target_feature_train_final = transformed_input_train_feature, target_feature_train_df
            input_feature_test_final, target_feature_test_final = transformed_input_test_feature, target_feature_test_df

            # concat train and test data.
            train_arr = np.c_[input_feature_train_final, target_feature_train_final]  # concat operation
            test_arr = np.c_[input_feature_test_final, target_feature_test_final]  # concat operation

            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # save object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise EnergyException(e, sys) from e



