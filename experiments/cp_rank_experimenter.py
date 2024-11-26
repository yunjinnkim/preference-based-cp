import joblib

joblib.parallel_config(backend="multiprocessing")

import joblib._store_backends
import sys

sys.path.insert(0, "/dss/dsshome1/04/ra43rid2/rank_cp/")

import random
from math import ceil, log2
import numpy as np
import torch
import openml
import mysql.connector
import types

import random
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from sklearn.model_selection import train_test_split
from simple_model import (
    ConformalRankingPredictor,
    ConformalPredictor,
    ClassifierModel,
    LabelRankingModel,
)
from sklearn.preprocessing import LabelEncoder

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

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


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

    X_train, X_test, y_train, y_test = train_test_split(
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
    alpha = parameters["alpha"]
    cal_size = parameters["fraction_cal_samples"]

    match parameters["model"]:
        # case "plnet":
        #     predictor = ConformalRankingPredictor(num_classes=y_train)
        case "random_forest":
            model = RandomForestClassifier(random_state=clf_seed)
            predictor = ConformalPredictor(model=model, alpha=alpha)
            predictor.fit(
                X_train,
                y_train,
                cal_size=parameters["fraction_cal_samples"],
                random_state=clf_seed,
            )

            gradient_updates = 0
        case "classifier_nn":
            model = ClassifierModel(
                input_dim=X_train.shape[1], hidden_dim=16, output_dim=num_classes
            )
            predictor = ConformalPredictor(model=model, alpha=alpha)
            predictor.fit(
                X_train,
                y_train,
                num_epochs=num_epochs,
                random_state=clf_seed,
                patience=num_epochs,
                batch_size=batch_size_clf,
                val_frac=val_frac,
                cal_size=cal_size,
                learning_rate=learning_rate,
            )
            gradient_updates = predictor.model.gradient_updates

        case "plnet_clf_arch":
            model = LabelRankingModel(
                input_dim=X_train.shape[1], hidden_dim=16, output_dim=num_classes
            )
            predictor = ConformalPredictor(model=model, alpha=alpha)
            predictor.fit(
                X_train,
                y_train,
                num_epochs=num_epochs,
                random_state=clf_seed,
                patience=num_epochs,
                batch_size=batch_size_clf,
                val_frac=val_frac,
                cal_size=cal_size,
                learning_rate=learning_rate,
            )
            gradient_updates = predictor.model.gradient_updates

        case "plnet":
            predictor = ConformalRankingPredictor(
                num_classes=num_classes, alpha=alpha, hidden_dim=16
            )
            effective_length = (1 - cal_size) * len(X_train)
            num_pairs = effective_length * (num_classes - 1)
            batch_size_rnk = num_pairs * batch_size_clf / effective_length
            batch_size_rnk = ceil(batch_size_rnk)
            predictor.fit(
                X_train,
                y_train,
                random_state=clf_seed,
                use_cross_isntance_data=False,
                num_epochs=num_epochs,
                patience=num_epochs,
                val_frac=val_frac,
                cal_size=cal_size,
                batch_size=batch_size_rnk,
                learning_rate=learning_rate,
            )
            gradient_updates = predictor.model.gradient_updates

        case "plnet_cross_instance":
            effective_length = (1 - cal_size) * len(X_train)
            num_pairs = (
                (num_classes - 1) * ceil(log2(effective_length)) * (effective_length)
            )
            num_pairs = ceil(num_pairs)
            batch_size_rnk = num_pairs * batch_size_clf / effective_length
            batch_size_rnk = ceil(batch_size_rnk)
            predictor = ConformalRankingPredictor(
                num_classes=num_classes, alpha=alpha, hidden_dim=16
            )
            predictor.fit(
                X_train,
                y_train,
                random_state=clf_seed,
                use_cross_isntance_data=True,
                num_epochs=num_epochs,
                patience=num_epochs,
                num_pairs=num_pairs,
                cal_size=cal_size,
                val_frac=val_frac,
                batch_size=batch_size_rnk,
                learning_rate=learning_rate,
            )
            gradient_updates = predictor.model.gradient_updates

    X_test = preprocessor.transform(X_test)

    # X_test = X_test.to_numpy()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray()
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.toarray()

    y_pred_crisp = predictor.model.predict(X_test)
    y_pred_set = predictor.predict_set(X_test)

    # Classifcation metrics of crisp (point) prediction
    score_f1 = f1_score(y_test, y_pred_crisp, average="macro")
    score_acc = accuracy_score(y_test, y_pred_crisp)
    score_bacc = balanced_accuracy_score(y_test, y_pred_crisp)
    # score_roc_auc = roc_auc_score(
    #     y_test, y_pred_crisp, average="macro", multi_class="ovr"
    # )

    # Conformal prediction metrics
    hits = [y_test[i] in y_pred_set[i] for i in range(len(y_test))]
    lens = [len(y_pred_set[i]) for i in range(len(y_test))]
    coverage_mean = np.mean(hits)
    coverage_std = np.std(hits)
    efficiency_mean = np.mean(lens)
    efficiency_std = np.std(lens)

    result_processor.process_results(
        {
            "score_f1": score_f1,
            "score_acc": score_acc,
            "score_bacc": score_bacc,
            # "score_roc_auc": score_roc_auc,
            "coverage_mean": coverage_mean,
            "coverage_std": coverage_std,
            "efficiency_mean": efficiency_mean,
            "efficiency_std": efficiency_std,
            "clf_seed": clf_seed,
            "mccv_seed": mccv_split_seed,
            "gradient_updates": gradient_updates,
        }
    )


if __name__ == "__main__":

    print(sys.version)

    experimenter = PyExperimenter(
        experiment_configuration_file_path="./config/config.yml"
    )

    # experimenter.db_connector.connect = types.MethodType(
    #     connect, experimenter.db_connector
    # )

    # experimenter.db_connector._start_transaction = types.MethodType(
    #     _start_transaction, experimenter.db_connector
    # )

    experimenter.fill_table_from_config()

    experimenter.execute(max_experiments=-1, experiment_function=worker)
