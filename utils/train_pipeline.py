import json

from torch import optim
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.evaluate import evaluate
from utils.train import train


def pipeline(model, train_dataloader, valid_dataloader, labels):
    N_EPOCHS = 100

    optimizer = optim.Adam(model.parameters(), lr=2e-6, eps=1e-8)
    criterion = nn.BCELoss()

    best_valid_acc = 0.0
    best_valid_acc_7 = 0.0
    best_valid_prec = 0.0
    best_valid_f1 = 0.0
    best_valid_rec = 0.0
    result_report = {}
    result_matrix = {}

    for epoch in tqdm(range(N_EPOCHS)):
        train_loss, train_acc, train_prec, train_f1, train_rec, _ = train(
            model, train_dataloader, optimizer, criterion
        )
        (
            valid_loss,
            valid_acc,
            valid_acc_7,
            valid_prec,
            valid_f1,
            valid_rec,
            report,
            matrix,
        ) = evaluate(model, valid_dataloader, criterion)

        if valid_acc >= best_valid_acc:
            result_report = report
            result_matrix = matrix

        best_valid_acc = max(best_valid_acc, valid_acc)
        best_valid_acc_7 = max(best_valid_acc_7, valid_acc_7)
        best_valid_prec = max(best_valid_prec, valid_prec)
        best_valid_rec = max(best_valid_rec, valid_rec)
        best_valid_f1 = max(best_valid_f1, valid_f1)

    print(f"Best valid acc = {best_valid_acc}")
    print(f"Best valid acc 7 = {best_valid_acc_7}")
    print(f"Best valid prec = {best_valid_prec}")
    print(f"Best valid f1 = {best_valid_f1}")
    print(f"Best valid rec = {best_valid_rec}")
    print(
        f"Classification report = \n {json.dumps(result_report, sort_keys=True, indent=4)}"
    )

    ax = plt.subplot()
    sns.heatmap(result_matrix, annot=True, fmt="g", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
