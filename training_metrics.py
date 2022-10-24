import json
import os
from argparse import ArgumentParser

class TrainingMetrics:
    def __init__(self, experiment):
        with open(os.path.join('results', experiment, 'trainer_state.json'), 'r') as j:
            stats = json.load(j)
        self.num_epochs = int(stats["epoch"])
        self.log = stats["log_history"]
    
    def get_losses(self):
        train_loss, val_loss = [], []
        for i in range(0, 2 * self.num_epochs, 2):
            train_loss.append(self.log[i]["loss"])
        for i in range(1, 2 * self.num_epochs, 2):
            val_loss.append(self.log[i]["eval_loss"]) 
        print("train / val loss by epoch:")
        for i in range(self.num_epochs):
            print(f"epoch {i + 1}: {train_loss[i]:.4f} / {val_loss[i]:.4f}")

    def get_val_metrics(self):
        val_f1, val_precision, val_recall, val_accuracy = [], [], [], []
        for i in range(1, 2 * self.num_epochs, 2):
            val_f1.append(self.log[i]["eval_f1"]) 
            val_precision.append(self.log[i]["eval_precision"]) 
            val_recall.append(self.log[i]["eval_recall"]) 
            val_accuracy.append(self.log[i]["eval_accuracy"]) 
        print("val metrics by epoch:")
        for i in range(self.num_epochs):
            print(f"epoch {i + 1}: f1 = {val_f1[i]:.4f}, precision = {val_precision[i]:.4f}, recall = {val_recall[i]:.4f}, accuracy = {val_accuracy[i]:.4f}")

if __name__ == "__main__":
    # parse command line arguments and load config
    parser = ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment to view', required=True)
    args = parser.parse_args()
    exp = args.exp
    metrics = TrainingMetrics(exp)
    metrics.get_losses()
    metrics.get_val_metrics()
