from energy_efficiency.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from energy_efficiency.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from energy_efficiency.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact
from energy_efficiency.entity.config_entity import ModelPusherConfig, ModelEvaluationConfig, ModelTrainerConfig
from energy_efficiency.exception import EnergyException
import sys
import os
from energy_efficiency.logger import logging
from energy_efficiency.components.data_ingestion import DataIngestion
from energy_efficiency.components.data_validation import DataValidation
from energy_efficiency.components.data_transformation import DataTransformation
from energy_efficiency.components.model_trainer import ModelTrainer
from energy_efficiency.components.model_evaluation import ModelEvaluation
from energy_efficiency.components.model_pusher import ModelPusher
from energy_efficiency.constant.training_pipeline import SAVED_MODEL_DIR  # /saved_models


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise EnergyException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise EnergyException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info("Starting data transformation")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise EnergyException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Starting model training")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"model training completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise EnergyException(e, sys)

    def start_model_evaluation(self, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info("Starting model evaluation")
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            logging.info(f"model evaluation completed and artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise EnergyException(e, sys)

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact):
        try:
            logging.info("Starting model pushing")
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info(f"model pushing completed and artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise EnergyException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting run_pipeline")
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            TrainPipeline.is_pipeline_running = False
            logging.info("run_pipeline is competed")
        except Exception as e:
            TrainPipeline.is_pipeline_running = False
            raise EnergyException(e, sys)
