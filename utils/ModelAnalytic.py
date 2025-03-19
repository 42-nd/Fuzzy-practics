import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    auc,
    roc_curve,
    precision_recall_curve,
)
import seaborn as sns
import shap
from typing import List
import catboost as cb


class ModelTester:
    """
    Утилитарный класс для оценки и визуализации качества модели классификации.

    Поддерживает модели CatBoost и scikit-learn (например, DecisionTreeClassifier).
    Вычисляет метрики (accuracy, balanced accuracy, precision, recall, F1, log_loss, roc_auc),
    строит графики ROC, Precision-Recall, распределение вероятностей и SHAP-анализ.

    Параметры:
      categorical: список категориальных признаков.
      numeric: список числовых признаков.
      target: название целевого признака.
    """

    def __init__(self, categorical: List[str], numeric: List[str], target: str):
        self.categorical = categorical
        self.numeric = numeric
        self.target = target

    def _prepare_data(
        self, model: ClassifierMixin, x_test: pd.DataFrame
    ) -> pd.DataFrame:
        # Для CatBoost используем Pool с указанием категориальных признаков.
        if hasattr(model, "get_feature_importance"):
            return cb.Pool(
                x_test[self.categorical + self.numeric], cat_features=self.categorical
            )
        else:
            return x_test[self.numeric + self.categorical]

    def metrics(self, model: ClassifierMixin, x_test: pd.DataFrame):
        """
        Вычисляет и выводит метрики качества модели, а также важность признаков.
        """
        data = self._prepare_data(model, x_test)
        y_pred = model.predict(data)
        y_prob = model.predict_proba(data)[:, 1]

        accuracy = accuracy_score(x_test[self.target], y_pred)
        balanced_accuracy = balanced_accuracy_score(x_test[self.target], y_pred)
        precision = precision_score(x_test[self.target], y_pred)
        recall = recall_score(x_test[self.target], y_pred)
        f1 = f1_score(x_test[self.target], y_pred)
        log_loss_value = log_loss(x_test[self.target], y_prob)
        roc_auc = roc_auc_score(x_test[self.target], y_prob)

        df_result = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Balanced Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "LogLoss",
                    "roc-auc",
                ],
                "Results": [
                    accuracy,
                    balanced_accuracy,
                    precision,
                    recall,
                    f1,
                    log_loss_value,
                    roc_auc,
                ],
            }
        )
        print(df_result)

        if hasattr(model, "get_feature_importance"):
            feat_imp = model.get_feature_importance()
            feat_names = model.feature_names_
        else:
            feat_imp = model.feature_importances_
            feat_names = self.categorical + self.numeric

        print("----------- Feature Importance -----------")
        for name, imp in sorted(
            zip(feat_names, feat_imp), key=lambda x: x[1], reverse=True
        ):
            print(f"{name}: {imp:.4f}")

    def plotter(self, model: ClassifierMixin, x_test: pd.DataFrame):
        """
        Строит ROC-кривую, Precision-Recall, распределение вероятностей и SHAP-анализ.
        """
        data = self._prepare_data(model, x_test)
        y_prob = model.predict_proba(data)[:, 1]

        fpr, tpr, _ = roc_curve(x_test[self.target], y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Кривая")
        plt.legend()
        plt.grid()
        plt.show()

        precision_vals, recall_vals, _ = precision_recall_curve(
            x_test[self.target], y_prob
        )
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid()
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.histplot(y_prob, bins=50, kde=True)
        plt.xlabel("Предсказанные вероятности")
        plt.title("Распределение предсказанных вероятностей")
        plt.show()

        # SHAP-анализ (для моделей деревьев используем TreeExplainer)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(x_test[self.categorical + self.numeric])
        shap.summary_plot(
            shap_values,
            x_test[self.categorical + self.numeric],
            plot_type="interaction",
        )

    def run(self, model: ClassifierMixin, x_test: pd.DataFrame):
        """
        Выполняет полный анализ: вычисление метрик и построение графиков.
        """
        self.metrics(model, x_test)
        self.plotter(model, x_test)
