from energy_efficiency.entity.artifact_entity import RegressionMetricArtifact
from energy_efficiency.exception import EnergyException
from sklearn.metrics import r2_score
import os
import sys


def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_r2_score = r2_score(y_true, y_pred)
        regression_metric = RegressionMetricArtifact(r2_score=model_r2_score)
        return regression_metric
    except Exception as e:
        raise EnergyException(e, sys)
