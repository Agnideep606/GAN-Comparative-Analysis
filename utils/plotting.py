import matplotlib.pyplot as plt
import os

def plot_loss_curves(losses, save_path, title="Training Loss"):
    """
    losses: dict { 'G': [..], 'D': [..] }
    """
    plt.figure(figsize=(8, 6))
    for label, loss in losses.items():
        plt.plot(loss, label=label)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics_dict, save_path, title="Evaluation Metrics"):
    """
    metrics_dict: { 'accuracy': 0.9, 'precision': 0.88, ... }
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(6, 4))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
