import numpy as np
import pandas as pd
import os
from datetime import datetime

# Modeling Framework - Sklearn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, f1_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

# ML Flow
import mlflow
import mlflow.sklearn
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.models.signature import infer_signature

# MLFlow Generates a lot of user warnings. Supressed for demo purposes.
import warnings
warnings.filterwarnings("ignore")
##
# You can aslo delete models via the API

def delete_model(model_name, versions=[], full_delete=False):

    client = mlflow.tracking.MlflowClient()

    for version in versions:
        client.delete_model_version(
            name=model_name, version=version
        )

    # Delete a registered model along with all its versions
    if full_delete:
        client.delete_registered_model(name=model_name)
        
    return None

##
# delete_model(model_name="WineQuality-Jim-GradientBoostingRegressor",  full_delete=True)
delete_model(model_name="EY-training-test",  full_delete=True)