"""Run many scikit-learn models x feature subsets x hyper-parameters in parallel and keep the best."""
from fitfleet.classification import Classification
from fitfleet.feature_research import FeatureResearch
from fitfleet.model import SavedModel, load
from fitfleet.regression import Regression
from fitfleet.tracking import ExecutionLog, ExecutionTracker
from fitfleet.utils import (FilteringCriteriaClassification, FilteringCriteriaRegression, ReportFormat, Sound,
                            ValidationType)

__version__ = "0.3.0"

__all__ = ["Classification", "Regression", "FeatureResearch", "FilteringCriteriaClassification", "FilteringCriteriaRegression",
           "ValidationType", "ReportFormat", "Sound", "SavedModel", "load", "ExecutionTracker", "ExecutionLog", "__version__"]
