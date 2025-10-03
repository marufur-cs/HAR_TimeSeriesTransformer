import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from load_model import get_model
from data_provide import get_dataset
from evaluate import evaluate_model_metrics
import numpy as np

# Create the parser
parser = argparse.ArgumentParser(description="Human Activity Recognition")

# Define dataset arguments
parser.add_argument('--dataset_name', type=str, help="Dataset name")
parser.add_argument('--no_features', type=int, default=10)
parser.add_argument('--no_classes', type=int, default=7)
parser.add_argument('--window_size', type=int, default=128, help="Number of observations in one sample")
parser.add_argument('--step_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)

# Mamba arguments
parser.add_argument('--model_name', type=str, help="Model name")
parser.add_argument('--hidden_dim', type=int, default=64, help="Neurons in the hidden layers of classificaiton head")
parser.add_argument('--experiments', type=int, default=1, help="Number of experiment to perform")

# iTransformer argumets
parser.add_argument('--emb_dim', type=int, default=128) # window size --> embedding dimension
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=10)

# PatchTST argumets
parser.add_argument('--pred_len', type=int, default=128, help='output from TST block--> B N pred_len')
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Define training arguments
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.01, help="Argument for AdamW")
parser.add_argument('--epochs', type=int, default=100)

# Parse the arguments
arg = parser.parse_args()

train_loader, test_loader = get_dataset(arg)


if arg.dataset_name =="har70+":
    if arg.model_name == 'itransformer':
            result_path="/home/rahmm224/AIinHealthProject/results/itransformer/har70+/"
    
    if arg.model_name == 'mamba':
            result_path="/home/rahmm224/AIinHealthProject/results/mamba/har70+/"

    if arg.model_name == 'patchtst':
            result_path="/home/rahmm224/AIinHealthProject/results/patchtst/har70+/"


acc = []
pre = []
rec = []
f1s = []
max_f1 = 0
print("Experiment started")
for e in range(arg.experiments):
    print(f"************ Experiment {e} ***************")
    model = get_model(arg)
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    loss_curve=[]
    # Training loop
    for epoch in range(arg.epochs):
        print(f"Epoch {epoch+1}/{arg.epochs}")
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to('cuda'), yb.to('cuda')
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # scheduler.step()
        total_loss /= len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        loss_curve.append(total_loss)
    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve, label='Train Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path+"loss.png")

    a, p, r, f1 = evaluate_model_metrics(model, test_loader, arg, max_f1)

    acc.append(a)
    pre.append(p)
    rec.append(r)
    f1s.append(f1)
    if f1>max_f1:
         max_f1=f1
    # Print metrics
    print(f"************ Experiment {e} done ***************")
with open(result_path+'output.txt', 'w') as file:
    print(f"Accuracy:  {np.mean(acc) * 100:.2f} ± {np.std(acc): .2f}%", file=file)
    print(f"Precision:  {np.mean(pre) * 100:.2f} ± {np.std(pre): .2f}%", file=file)
    print(f"Recall:  {np.mean(rec) * 100:.2f} ± {np.std(rec): .2f}%", file=file)
    print(f"F1 Score:  {np.mean(f1s) * 100:.2f} ± {np.std(f1s): .2f}%", file=file)