import argparse
import datetime

import torch
import torch.autograd
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import models
from dataset import get_prepared_dataset, split

file_path = '/Users/silence/Workbench/ml/ml_labs/LegacyData/half.csv'
columns_count = 60000
target_column = 'first_label'
useless_columns = ['second_label', 'third_label']
names_array = ['gender', 'age', 'first_label', 'second_label', 'third_label',
               *['c_{}'.format(i + 1) for i in range(columns_count)]]


def evaluate(model, test_loader, criterion):
    # production mode to NN (batch_norm to statistics (mean, divergence), dropout off)
    model.eval()
    # disable gradient
    with torch.no_grad():
        avg_loss = 0.0
        val_acc = 0
        for x, y in test_loader:
            # forward
            out = model(x)
            # calcilate loss
            loss = criterion(out, y)
            # 1 means axis 1
            _, pred = torch.max(out.data, 1)
            val_acc += (pred == y).numpy().mean()
            avg_loss += loss.item()
        avg_loss /= len(test_loader)
        val_acc /= len(test_loader)
    model.train()
    return avg_loss, val_acc


def train(args):
    writer = SummaryWriter(f'./logs/{args.type}-{datetime.datetime.now()}')
    X, y = get_prepared_dataset(file_path, names_array, target_column, useless_columns,
                                augmentation_multiplier=args.multiplier,
                                augmentation_slice_size=args.slice)

    train_loader, test_loader = split(X, y, test_size=0.2, batch_size=args.batch, random_state=13)
    net = getattr(models, args.type)(args.num_classes, X.shape[1])

    # batch size, channel count, height, width
    # writer.add_graph(net, torch.zeros(1, 1, 28, 28))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    for e in range(args.epochs):
        net.train()
        for i, (x, y) in enumerate(train_loader):
            # flush gradient
            optimizer.zero_grad()
            # forward
            out = net(x)
            # calculate loss
            loss = criterion(out, y)
            _, pred = torch.max(out.data, 1)
            acc = pred.eq(y).sum().item() / y.size(0)  # (pred == y).numpy().mean()
            # backward
            loss.backward()
            optimizer.step()
            # writer.add_scalar('Train/LR', scheduler.get_lr()[0], e * len(train_loader) + i)
            if i % args.print_every == 0:
                val_loss, val_acc = evaluate(net, test_loader, criterion)
                print(f'Epoch: {e}, Iteration: {i} \n'
                      f'X: {x.size()}. Y: {y.size()} \n'
                      f'Loss: {loss.item()} , Acc: {acc} \n'
                      f'Val Loss: {val_loss}, Val Acc: {val_acc} \n'
                      f'-----------------------------------------\n'
                      )
                writer.add_scalar('Train/Acc', acc, e * len(train_loader))
                writer.add_scalar('Train/Loss', loss.item(), e * len(train_loader) + i)
                writer.add_scalar('Val/Loss', val_loss, e * len(train_loader) + i)  # last arg is global iterator
                writer.add_scalar('Val/Acc', val_acc, e * len(train_loader) + i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script of ECG problem.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=80, help='Total number of epochs.')
    parser.add_argument('--batch', type=int, default=10000, help='Batch size.')
    parser.add_argument('--slice', type=int, default=1000, help='Wide of augmentation window.')
    parser.add_argument('--multiplier', type=int, default=20, help='Number of repeats of augmentation process.')
    parser.add_argument('--print_every', type=int, default=1, help='Print every # iterations.')
    parser.add_argument('--num_classes', type=int, default=9, help='Num classes.')
    parser.add_argument('--type', choices=['CNN', 'MLP'], default='MLP', help='Type of Network')
    args = parser.parse_args()

    train(args)
