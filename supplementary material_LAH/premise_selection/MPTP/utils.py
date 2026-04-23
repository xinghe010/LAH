import json
import pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def dumps_list_to_json(obj, file_path):
    with open(file_path, "w+") as f:
        f.write("\n".join([json.dumps(element) for element in obj]))

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def dump_pickle_file(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def set_recorder(name, logfile):
    recorder = logging.getLogger(name)
    recorder.setLevel(logging.INFO)
    rf_handler = logging.StreamHandler()
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    recorder.addHandler(rf_handler)
    recorder.addHandler(f_handler)
    return recorder

class Statements:
    def __init__(self, statements_file):
        self.statements = self.build_statements(statements_file)

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, name):
        return self.statements[name]

    def __iter__(self):
        return self.statements.__iter__()

    def build_statements(self, statements_file):
        statements = dict()
        lines = read_file(statements_file)
        for line in lines:
            name = line.split(',')[0].replace("fof(", "")
            statements[name] = line.replace(" ", "")
        return statements

def py_plot(title,
            train_loss, valid_loss,
            train_acc, valid_acc,
            train_f1=None, valid_f1=None,
            train_precision=None, valid_precision=None,
            save_file="figure.png"):

    assert len(train_loss) == len(valid_loss)
    assert len(train_acc) == len(valid_acc)

    colors = {
        'train': '#D62728',
        'valid': '#1F77B4',
        'train_light': '#FF9896',
        'valid_light': '#AEC7E8'
    }

    line_style = {
        'linewidth': 2.5,
        'alpha': 0.9,
        'marker': 'o',
        'markersize': 5,
        'markeredgewidth': 1,
        'markerfacecolor': 'white'
    }

    plt.figure(figsize=(14, 10))
    plt.suptitle(title, fontsize=16, y=1.02, fontweight='bold')
    epochs = np.arange(1, len(train_loss) + 1)

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_loss, color=colors['train'], label='Train Loss', **line_style)
    ax1.plot(epochs, valid_loss, color=colors['valid'], label='Valid Loss', **line_style)
    ax1.set_title('Training & Validation Loss', pad=15, fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(frameon=True, framealpha=0.9, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, train_acc, color=colors['train'], label='Train Accuracy', **line_style)
    ax2.plot(epochs, valid_acc, color=colors['valid'], label='Valid Accuracy', **line_style)
    ax2.set_title('Training & Validation Accuracy', pad=15, fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(frameon=True, framealpha=0.9, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.4)

    if train_f1 is not None and valid_f1 is not None:
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(epochs, train_f1, color=colors['train'], label='Train F1', **line_style)
        ax3.plot(epochs, valid_f1, color=colors['valid'], label='Valid F1', **line_style)
        ax3.set_title('F1 Score', pad=15, fontsize=14)
        ax3.set_xlabel('Epochs', fontsize=11)
        ax3.set_ylabel('F1 Score', fontsize=11)
        ax3.legend(frameon=True, framealpha=0.9, loc='lower right')
        ax3.grid(True, linestyle='--', alpha=0.4)

    if train_precision is not None and valid_precision is not None:
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(epochs, train_precision, color=colors['train'], 
                label='Train Precision', **line_style)
        ax4.plot(epochs, valid_precision, color=colors['valid'], 
                label='Valid Precision', **line_style)
        ax4.set_title('Precision', pad=15, fontsize=14)
        ax4.set_xlabel('Epochs', fontsize=11)
        ax4.set_ylabel('Precision', fontsize=11)
        ax4.legend(frameon=True, framealpha=0.9, loc='lower right')
        ax4.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    base_name = os.path.splitext(save_file)[0]
    plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
def plot_metrics(history, save_path, marker_interval=5):

    epochs = np.arange(1, len(history['train_acc']) + 1)

    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.edgecolor': 'black',  
        'axes.linewidth': 1.2,
        'axes.facecolor': 'white', 
        'figure.facecolor': 'white', 
        'savefig.facecolor': 'white' 
    })

    metrics = {
        'Accuracy': 'acc',
        'F1': 'f1',
        'Recall': 'recall',
        'Precision': 'precision'
    }
    styles = {
        'acc': {'color': 'b', 'marker': 'o', 'ls': '-'},
        'f1': {'color': 'y', 'marker': 's', 'ls': '--'},
        'recall': {'color': 'g', 'marker': '^', 'ls': '--'},
        'precision': {'color': 'r', 'marker': 'D', 'ls': '-'}
    }

    def get_ylim(phase):
        all_values = []
        for metric_key in metrics.values():
            all_values.extend(history[f'{phase}_{metric_key}'])
        buffer = (max(all_values) - min(all_values)) * 0.1
        return max(0.7, min(all_values) - buffer), min(1.0, max(all_values) + buffer)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=600)

    ax1.set_title('the Result on Train set', pad=12)
    ax2.set_title('the Result on Valid set', pad=12)

    for metric_name, metric_key in metrics.items():
        y_train = history[f'train_{metric_key}']
        style = styles[metric_key]

        ax1.plot(epochs, y_train,
                 color=style['color'],
                 linestyle=style['ls'],
                 linewidth=1.5)

        marker_epochs = epochs[::marker_interval]
        marker_values = y_train[::marker_interval]
        ax1.plot(marker_epochs, marker_values,
                 linestyle='',
                 marker=style['marker'],
                 color=style['color'],
                 markersize=6,
                 label=metric_name)

    ax1.set_ylim(get_ylim('train'))
    ax1.legend(loc='lower right', frameon=True, ncol=1)

    for metric_name, metric_key in metrics.items():
        y_valid = history[f'valid_{metric_key}']
        style = styles[metric_key]

        ax2.plot(epochs, y_valid,
                 color=style['color'],
                 linestyle=style['ls'],
                 linewidth=1.5)

        marker_epochs = epochs[::marker_interval]
        marker_values = y_valid[::marker_interval]
        ax2.plot(marker_epochs, marker_values,
                 linestyle='',
                 marker=style['marker'],
                 color=style['color'],
                 markersize=6,
                 label=metric_name)

    ax2.set_ylim(get_ylim('valid'))
    ax2.legend(loc='lower right', frameon=True, ncol=1)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()
