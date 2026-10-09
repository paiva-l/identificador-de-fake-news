"""Gráficos calculados a partir das previsões da execução atual."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, precision_recall_curve, average_precision_score)

LABELS = [0, 1]
NAMES = ["Fake (0)", "Real (1)"]

def save_show(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight", dpi=180)
    plt.show()
    plt.close(fig)

def plot_class_distribution(y_train, y_test, directory):
    counts = pd.DataFrame({"Treino": pd.Series(y_train).value_counts(),
                           "Teste": pd.Series(y_test).value_counts()}).reindex(LABELS, fill_value=0)
    counts.index = NAMES
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, rot=0, color=["#2579a8", "#d89233"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    ax.set(title="Distribuição das classes por partição", xlabel="Classe", ylabel="Notícias")
    ax.margins(y=0.15)
    save_show(fig, directory, "01_distribuicao_classes")

def fake_scores(model, X):
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError("Gráficos exigem classes [0=Fake, 1=Real].")
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, classes.index(0)]
    # Na classificação binária, a margem positiva favorece classes_[1].
    return -model.decision_function(X)

def plot_evaluation(models, X_test, y_test, directory):
    y = np.asarray(y_test)
    if set(y) != {0, 1}:
        raise ValueError("Avaliação gráfica exige ambas as classes no teste.")
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    for suffix, normalized in [("contagens", False), ("percentuais", True)]:
        fig, axes = plt.subplots(len(models), 1, figsize=(6, 4*len(models)), squeeze=False)
        for ax, (name, pred) in zip(axes.ravel(), predictions.items()):
            matrix = confusion_matrix(y, pred, labels=LABELS, normalize="true" if normalized else None)
            sns.heatmap(matrix, annot=True, fmt=".1%" if normalized else "d", cmap="Blues",
                        vmin=0, vmax=1 if normalized else None, cbar=False,
                        xticklabels=NAMES, yticklabels=NAMES, ax=ax)
            ax.set(title=f"{name} — {'% dentro da classe real' if normalized else 'contagens'}",
                   xlabel="Classe predita", ylabel="Classe real")
        save_show(fig, directory, f"02_matrizes_{suffix}")

    reports = {}
    for name, pred in predictions.items():
        report = classification_report(y, pred, labels=LABELS, target_names=NAMES,
                                       output_dict=True, zero_division=0)
        reports[name] = {metric: report["macro avg"][metric] for metric in ["precision", "recall", "f1-score"]}
        reports[name]["accuracy"] = float((y == pred).mean())
    fig, ax = plt.subplots(figsize=(10, max(3, len(models)*1.2)))
    sns.heatmap(pd.DataFrame(reports).T.rename(columns={"precision": "Precisão macro", "recall": "Recall macro",
                "f1-score": "F1 macro", "accuracy": "Acurácia"}),
                annot=True, fmt=".3f", vmin=0, vmax=1, cmap="YlGnBu", ax=ax)
    ax.set(title="Métricas no teste — mesma partição", xlabel="", ylabel="Modelo")
    save_show(fig, directory, "03_metricas_teste")

    # Fake é a classe de interesse; isto não altera o encoding do treinamento.
    y_fake = (y == 0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, model in models.items():
        scores = fake_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_fake, scores)
        precision, recall, _ = precision_recall_curve(y_fake, scores)
        axes[0].plot(fpr, tpr, label=f"{name} · AUC={roc_auc_score(y_fake, scores):.3f}")
        axes[1].plot(recall, precision, label=f"{name} · AP={average_precision_score(y_fake, scores):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Referência aleatória")
    axes[1].axhline(y_fake.mean(), linestyle="--", color="gray", label=f"Prevalência Fake={y_fake.mean():.1%}")
    axes[0].set(title="ROC — classe de interesse Fake", xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
    axes[1].set(title="Precisão–recall — classe de interesse Fake", xlabel="Recall", ylabel="Precisão")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.legend(loc="best", fontsize=8)
    save_show(fig, directory, "04_roc_e_precisao_recall")

    errors = pd.DataFrame({name: {"Fake → Real": int(((y == 0) & (pred == 1)).sum()),
                                 "Real → Fake": int(((y == 1) & (pred == 0)).sum())}
                           for name, pred in predictions.items()}).T
    fig, ax = plt.subplots(figsize=(10, 5))
    errors.plot.barh(ax=ax, color=["#c84b47", "#d89233"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    ax.set(title="Erros por direção no teste", xlabel="Número de notícias", ylabel="Modelo")
    ax.margins(x=0.2)
    save_show(fig, directory, "05_erros_por_direcao")

def plot_times(benchmark, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title in zip(axes, ["Tempo Treino (s)", "Tempo Predição (s)"],
                               ["Tempo de ajuste do classificador", "Predição do lote de teste"]):
        values = benchmark.set_index("Modelo")[column]
        values.plot.barh(ax=ax, color="#2579a8")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.5f", padding=3)
        ax.set(title=title, xlabel="Segundos — exclui construção das features", ylabel="")
        ax.margins(x=0.3)
    save_show(fig, directory, "06_tempos")
