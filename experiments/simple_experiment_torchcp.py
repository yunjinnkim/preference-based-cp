import joblib
import torchcp

joblib.parallel_config(backend="multiprocessing")

import joblib._store_backends
import sys

sys.path.insert(0, "/dss/dsshome1/04/ra43rid2/conformal_prediction_via_ranking/")

import random
from math import ceil, log2
import numpy as np
import torch

torch.set_default_device("cpu")
import openml
import mysql.connector
import types

import random
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from sklearn.model_selection import train_test_split
from models.classifier_model import ClassifierModel
from models.ranking_models import LabelRankingModel
from sklearn.preprocessing import LabelEncoder
from util.ranking_datasets import TabularDataset

from torch.utils.data import DataLoader

from conformal.conformal import ConformalPredictor

from torchcp.classification.score import APS, THR, TOPK, RAPS, SAPS, Margin, KNN
from conformal.conformal import IDENTITY, OWN_APS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification import Metrics

from py_experimenter.database_connector_mysql import (
    DatabaseConnectorMYSQL,
    DatabaseConnector,
)


def connect(self):

    db = mysql.connector.connect(
        # TODO enter DB credentials
    )
    return db


def _start_transaction(self, connection, readonly=False):
    if not readonly:
        connection.start_transaction()


DatabaseConnectorMYSQL.connect = connect
DatabaseConnectorMYSQL._start_transaction = _start_transaction

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from itertools import product

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from torchcp.classification.utils import Metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)


def evaluate(
    predictor,
    model,
    alpha,
    conformity_score_name,
    test_loader,
    num_classes,
    clf_seed,
    mccv_split_seed,
    gradient_updates,
    result_processor,
    len_train,
    len_test,
    len_cal,
):

    predictions_sets_list = []
    predictions_list = []
    labels_list = []
    logits_list = []
    feature_list = []

    # Evaluate in inference mode
    predictor._model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device and get predictions
            inputs = batch[0]
            labels = batch[1]

            # Get predictions as bool tensor (N x C)
            if isinstance(predictor, ConformalPredictor):
                batch_predictions = predictor.predict(inputs, alpha=alpha)
            else:
                batch_predictions = predictor.predict(inputs)

            logits = model(inputs)

            predicted_label = logits.argmax(axis=1)
            # Accumulate predictions and labels
            predictions_sets_list.append(batch_predictions)
            predictions_list.append(predicted_label)
            labels_list.append(labels)
            logits_list.append(logits)
            feature_list.append(inputs)

        # Concatenate all batches
        val_prediction_sets = torch.cat(predictions_sets_list, dim=0)  # (N_val x C)
        val_predictions = torch.cat(predictions_list, dim=0)
        val_labels = torch.cat(labels_list, dim=0)  # (N_val,)
        val_logits = torch.cat(logits_list, dim=0)
        val_features = torch.cat(feature_list, dim=0)

        y_pred = val_predictions.detach().cpu().numpy()
        y_true = val_labels.detach().cpu().numpy()
        # Compute evaluation metrics
        metric = Metrics()

        metrics = {
            "coverage_rate": metric("coverage_rate")(
                prediction_sets=val_prediction_sets, labels=val_labels
            ),
            "average_size": metric("average_size")(
                prediction_sets=val_prediction_sets, labels=val_labels
            ),
            "cov_gap": metric("CovGap")(
                prediction_sets=val_prediction_sets,
                labels=val_labels,
                alpha=alpha,
                num_classes=num_classes,
            ),
            "vio_classes": metric("VioClasses")(
                prediction_sets=val_prediction_sets,
                labels=val_labels,
                alpha=alpha,
                num_classes=num_classes,
            ),
            "sscv": metric("SSCV")(
                prediction_sets=val_prediction_sets,
                labels=val_labels,
                alpha=alpha,
            ),
            "wsc": metric("WSC")(
                val_features,
                prediction_sets=val_prediction_sets,
                labels=val_labels,
            ),
            "acc": accuracy_score(y_true, y_pred),
            "bacc": balanced_accuracy_score(y_true, y_pred),
        }

        metrics.update(
            {
                "conformity_score": conformity_score_name,
                "alpha": alpha,
                "gradient_updates": gradient_updates,
                "clf_seed": clf_seed,
                "mccv_seed": mccv_split_seed,
                "len_train": len_train,
                "len_test": len_test,
                "len_cal": len_cal,
            }
        )

    result_processor.process_logs({"results": metrics})


def worker(parameters: dict, result_processor: ResultProcessor, custom_config: dict):

    master_seed = parameters["master_seed"]
    rng = random.Random(master_seed)
    mccv_split_seed = rng.randint(0, 2**32 - 1)
    clf_seed = rng.randint(0, 2**32 - 1)

    random.seed(clf_seed)
    np.random.seed(clf_seed)
    torch.manual_seed(clf_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(clf_seed)
        torch.cuda.manual_seed_all(clf_seed)

    dataset = openml.datasets.get_dataset(parameters["openml_id"])
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    # Automatically identify categorical and numerical columns
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_classes = len(np.unique(y))

    X_train, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=mccv_split_seed
    )

    # Encode labels
    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Preprocessing for numerical data: Impute missing values, then scale
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data: Impute missing values, then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train = preprocessor.fit_transform(X_train)

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.toarray()
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.toarray()

    batch_size_clf = 32
    val_frac = 0.2
    num_epochs = 300
    learning_rate = 0.01
    cal_size = parameters["fraction_cal_samples"]
    alphas = [0.05, 0.1, 0.2]

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train, y_train, test_size=cal_size
    )

    match parameters["model"]:

        case "classifier":
            model = ClassifierModel(
                input_dim=X_train.shape[1], hidden_dims=[20, 20], output_dim=num_classes
            )
            # model.cuda()
            model.fit(
                X_train,
                y_train,
                num_epochs=num_epochs,
                random_state=clf_seed,
                patience=num_epochs,
                batch_size=batch_size_clf,
                val_frac=val_frac,
                learning_rate=learning_rate,
            )

            X_test = preprocessor.transform(X_test_orig)

            # X_test = X_test.to_numpy()
            if not isinstance(X_test, np.ndarray):
                X_test = X_test.toarray()
            if not isinstance(y_test, np.ndarray):
                y_test = y_test.toarray()

            ds_test = TabularDataset(X_test, y_test)
            ds_cal = TabularDataset(X_cal, y_cal)
            test_loader = DataLoader(ds_test)
            cal_loader = DataLoader(ds_cal)

            len_train = len(X_train)
            len_test = len(ds_test)
            len_cal = len(ds_cal)

            names = ["rand_aps", "aps", "thr", "topk", "raps", "saps", "margin"]
            conformity_scores = [
                APS(randomized=True),
                APS(randomized=False),
                THR(),
                TOPK(),
                RAPS(),
                SAPS(),
                Margin(),
            ]
            for (name, conformity_score), alpha in product(
                zip(names, conformity_scores), alphas
            ):
                predictor = SplitPredictor(conformity_score, model)

                gradient_updates = predictor._model.gradient_updates

                predictor.calibrate(cal_loader, alpha)

                evaluate(
                    predictor,
                    model,
                    alpha,
                    name,
                    test_loader,
                    num_classes,
                    clf_seed,
                    mccv_split_seed,
                    gradient_updates,
                    result_processor,
                    len_train,
                    len_test,
                    len_cal,
                )

            result_processor.process_results(
                {"mccv_seed": mccv_split_seed, "clf_seed": clf_seed}
            )

        case "ranker":
            model = LabelRankingModel(
                input_dim=X_train.shape[1], hidden_dims=[20, 20], output_dim=num_classes
            )
            model.fit(
                X_train,
                y_train,
                num_epochs=num_epochs,
                random_state=clf_seed,
                patience=num_epochs,
                batch_size=batch_size_clf,
                val_frac=val_frac,
                learning_rate=learning_rate,
            )
            # predictor = ConformalPredictor(model)
            # predictor.fit(X_cal, y_cal)

            predictor_vanilla = SplitPredictor(IDENTITY(), model)
            predictor_own_aps = SplitPredictor(OWN_APS(randomized=False), model)
            predictor_own_randomized_aps = SplitPredictor(
                OWN_APS(randomized=True), model
            )

            predictors = [
                predictor_vanilla,
                predictor_own_aps,
                predictor_own_randomized_aps,
            ]
            names = ["ranker_vanilla", "ranker_own_aps", "ranker_own_rand_aps"]

            X_test = preprocessor.transform(X_test_orig)

            # X_test = X_test.to_numpy()
            if not isinstance(X_test, np.ndarray):
                X_test = X_test.toarray()
            if not isinstance(y_test, np.ndarray):
                y_test = y_test.toarray()

            ds_test = TabularDataset(X_test, y_test)
            ds_cal = TabularDataset(X_cal, y_cal)

            len_train = len(X_train)
            len_test = len(ds_test)
            len_cal = len(ds_cal)

            test_loader = DataLoader(ds_test)
            cal_loader = DataLoader(ds_cal)
            for (name, predictor), alpha in product(zip(names, predictors), alphas):
                gradient_updates = predictor_vanilla._model.gradient_updates
                predictor.calibrate(cal_loader, alpha)

                evaluate(
                    predictor,
                    model,
                    alpha,
                    name,
                    test_loader,
                    num_classes,
                    clf_seed,
                    mccv_split_seed,
                    gradient_updates,
                    result_processor,
                    len_train,
                    len_test,
                    len_cal,
                )

            result_processor.process_results(
                {"mccv_seed": mccv_split_seed, "clf_seed": clf_seed}
            )


if __name__ == "__main__":

    print(sys.version)

    experimenter = PyExperimenter(
        experiment_configuration_file_path="./config/cfg_simple_debug.yml",
        use_codecarbon=False,
    )

    # experimenter.db_connector.connect = types.MethodType(
    #     connect, experimenter.db_connector
    # )

    # experimenter.db_connector._start_transaction = types.MethodType(
    #     _start_transaction, experimenter.db_connector
    # )

    experimenter.fill_table_from_config()

    experimenter.execute(max_experiments=-1, experiment_function=worker)
