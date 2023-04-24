
from datetime import datetime
import os
from energy_efficiency.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME  # energy_pipeline
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)  # artifact  => timestamp
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )  # artifact/data_ingestion
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )  # artifact/data_ingestion/feature_store/energy.csv
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )  # artifact/data_ingestion/ingested/train.csv
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )  # artifact/data_ingestion/ingested/test.csv
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO  # 0.2
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME  # energy_efficiency


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)  # artifact/data_validation
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)  # data_validation/validated
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)  # data_validation/invalid
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)  # validated/train.csv
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)  # validated/test.csv
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)  # invalid/train.csv
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)  # invalid/test.csv
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )  # data_validation/drift_report/report.yaml


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        # artifact/data_transformation
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"))
        # data_transformation/transformed/train.npy
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"))
        # data_transformation/transformed/test.npy
        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)
        # data_transformation/transformed_object/preprocessing.pkl


class ModelTrainerConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME  # artifact/model_trainer
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,  # model_trainer/trained_model/model.pkl
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE  # 0.6
        self.over_fitting_under_fitting_threshold: float = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD  # 0.05
        self.learning_rate: float = training_pipeline.MODEL_TRAINER_LEARNING_RATE  # 0.1
        self.n_estimators: int = training_pipeline.MODEL_TRAINER_N_ESTIMATORS   # 1000
        self.max_depth: int = training_pipeline.MODEL_TRAINER_MAX_DEPTH  # 8
        self.col_sample_by_tree: float = training_pipeline.MODEL_TRAINER_COL_SAMPLE_BY_TREE  # 0.4


class ModelEvaluationConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUATION_DIR_NAME)  # artifact/model_evaluation
        self.report_file_path = os.path.join(self.model_evaluation_dir, training_pipeline.MODEL_EVALUATION_REPORT_NAME)  # model_evaluation/report.yaml
        self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE  # 0.02


class ModelPusherConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_pusher_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_PUSHER_DIR_NAME)  # artifact/model_pusher
        self.model_file_path = os.path.join(self.model_pusher_dir, training_pipeline.MODEL_FILE_NAME)  # model_pusher/model.pkl
        timestamp = round(datetime.now().timestamp())
        self.saved_model_path = os.path.join(training_pipeline.SAVED_MODEL_DIR, f"{timestamp}", training_pipeline.MODEL_FILE_NAME)  # saved_model/timestamp/model.pkl
