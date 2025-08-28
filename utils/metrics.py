import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            preds = model(x).argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
