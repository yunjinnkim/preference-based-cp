import numpy as np

from models.ranking_models import LabelRankingModel


class ConformalPredictor:

    def __init__(self, model):
        self._model = model

    def fit(self, X_cal, y_cal, **kwargs):
        self.X_cal = X_cal
        self.y_cal = y_cal
        if isinstance(self._model, LabelRankingModel):
            y_pred_cal = self._model.predict_class_skills(self.X_cal)
        else:
            y_pred_cal = self._model.predict_proba(self.X_cal)
        self.scores = 1 - y_pred_cal[np.arange(len(self.y_cal)), self.y_cal]

    def predict_set(self, X, alpha=0.2):
        n = len(self.scores)
        self.threshold = np.quantile(
            self.scores,
            np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1),
            method="inverted_cdf",
        )
        if isinstance(self._model, LabelRankingModel):
            y_preds = self._model.predict_class_skills(X)
        else:
            y_preds = self._model.predict_proba(X)
        pred_sets = []
        for y_pred in y_preds:
            pred_set = np.where(1 - y_pred <= self.threshold)[0]
            if pred_set.size == 0:
                pred_set = [np.argmax(y_pred)]
            pred_sets.append(pred_set)
        return pred_sets

    def predict(self, X, alpha=0.2):
        n = len(self.scores)
        self.threshold = np.quantile(
            self.scores,
            np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1),
            method="inverted_cdf",
        )
        y_pred = self._model(X)
        return 1 - y_pred <= self.threshold

    # def predict(self, X, alpha=0.2):
    #     y_crisp = self._model.predict(X)
    #     y_set = self.predict_set(X, alpha=alpha)
    #     return y_crisp, y_set


class ConformalRankingPredictor:
    def __init__(self, num_classes, alpha=0.05, hidden_dim=16):
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.alpha = alpha

    def fit(
        self,
        X,
        y,
        random_state,
        num_epochs=100,
        cal_size=0.33,
        use_cross_isntance_data=False,
        num_pairs=-1,
        **kwargs,
    ):
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=cal_size)

        self.model = LabelRankingModel(
            input_dim=X_train.shape[1] + y.max() + 1, hidden_dim=self.hidden_dim
        )
        self.model.fit(
            X_train,
            y_train,
            use_cross_instance_dataset=use_cross_isntance_data,
            num_pairs=num_pairs,
            random_state=random_state,
            num_epochs=num_epochs,
            **kwargs,
        )

        # here we usually compute non conformity scores. For the ranker
        # we use the predicted latent skill value

        # y_pred_cal = self.model.predict_proba(X_cal)
        # self.scores = 1 - y_pred_cal[np.arange(len(y_cal)), y_cal]
        cal_dyads = create_dyads(X_cal, y_cal, self.num_classes)
        with torch.no_grad():
            self.scores = -self.model(cal_dyads).detach().cpu().numpy()
        n = len(self.scores)
        # TODO check alpha here
        self.threshold = np.quantile(
            self.scores,
            np.clip(np.ceil((n + 1) * (1 - self.alpha)) / n, 0, 1),
            method="inverted_cdf",
        )

    def predict_set(self, X):

        y_skills = self.model.predict_class_skills(X)

        pred_sets = []
        for y_skill in y_skills:
            pred_set = np.where(-y_skill <= self.threshold)[0]
            pred_sets.append(pred_set)
        return pred_sets
