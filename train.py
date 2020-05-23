import argparse
import datetime
import os

import tpot

import utils
import torch
import torch.autograd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

import models
import dataset
import test
import sklearn.model_selection


def _write_checkpoint(writer, e, epochs, i, iteration_per_epochs, acc, loss, val_acc, val_loss):
    in_epoch_progress = round(i / iteration_per_epochs, 2)
    full_progress = round((e * iteration_per_epochs + i) / (epochs * iteration_per_epochs), 2)
    print(f'{datetime.datetime.now()}\n '
          f'Epoch: {e}/{epochs}, Iteration: {i}/{iteration_per_epochs}, '
          f'In epoch progress ({in_epoch_progress * 100} %)\n'
          f'Full progress {full_progress * 100} %\n'
          f'Loss: {loss.item()} , Acc: {acc} \n'
          f'Val Loss: {val_loss}, Val Acc: {val_acc} \n'
          )
    step_label = e * iteration_per_epochs + i
    writer.add_scalar('Train/Acc', acc, step_label)
    writer.add_scalar('Train/Loss', loss.item(), step_label)
    writer.add_scalar('Val/Loss', val_loss, step_label)
    writer.add_scalar('Val/Acc', val_acc, step_label)
    return step_label


def train(args):
    writer = SummaryWriter(f'./logs/{args.type}-{datetime.datetime.now()}_batch={args.batch}_slice={args.slice}_mul={args.multiplier}')
    checkpoint_prefix = f'{args.type}_{datetime.datetime.now()}'
    print(f'{datetime.datetime.now()} Loading data')
    reference_path = f'{args.base_path}/REFERENCE.csv'
    df = dataset.Loader(args.base_path, reference_path).load_as_df_for_net(normalize=True)
    train_df, test_df = sklearn.model_selection.train_test_split(df, random_state=42, test_size=0.3)
    print(f'{datetime.datetime.now()} Create loaders')
    train_df = dataset.ECGDataset(train_df, slices_count=args.multiplier, slice_len=args.slice, random_state=42)
    test_df = dataset.ECGDataset(test_df, slices_count=args.multiplier, slice_len=args.slice, random_state=42)
    train_loader = DataLoader(train_df, batch_size=args.batch, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=args.batch, num_workers=4, shuffle=True)
    use_cuda = torch.cuda.is_available()

    if args.type.startswith('VGG_'):
        net = models.get_vgg(args.type.split('_')[-1], batch_norm=True)
    else:
        net = getattr(models, args.type)()
    if use_cuda:
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    print(f'{datetime.datetime.now()} Train started')
    iteration_per_epochs = len(train_loader)
    for e in range(args.epochs):
        net.train()
        acc = 0.0
        loss = 0.0
        for i, (non_ecg, ecg, y) in enumerate(train_loader):
            if use_cuda:
                non_ecg, ecg, y = non_ecg.cuda(), ecg.cuda(), y.cuda()
            optimizer.zero_grad()
            out = net(non_ecg, ecg)
            loss = criterion(out, y)
            _, pred = torch.max(out.data, 1)
            acc = pred.eq(y).sum().item() / y.size(0)  # (pred == y).numpy().mean()
            loss.backward()
            optimizer.step()
            if i % args.print_every == 0:
                val_loss, val_acc = test.evaluate(net, test_loader, criterion)
                _write_checkpoint(writer, e, args.epochs, i + 1, iteration_per_epochs, acc, loss, val_acc, val_loss)

        val_loss, val_acc = test.evaluate(net, test_loader, criterion)
        label = _write_checkpoint(writer, e, args.epochs, iteration_per_epochs,
                                  iteration_per_epochs, acc, loss, val_acc, val_loss)
        checkpoint_name = os.path.join(checkpoint_prefix, f'e_{e}_(step_{label}).pth')
        utils.save_net_model(net, checkpoint_name)
        print(f'Checkpoint of epoch {e} saved'
              f'-----------------------------------------\n'
              )


def draw(args):
    reference_path = f'{args.base_path}/REFERENCE.csv'
    df = dataset.Loader(args.base_path, reference_path).load_as_df_for_net(normalize=True)
    sample = df.iloc[0]['ecg']
    labels = list(range(sample.shape[1]))
    for i in range(sample.shape[0]):
        plt.plot(labels, sample[i], label=f'sensor_{i}')
    plt.grid()
    plt.legend()
    plt.show()


def train_ml(args):
    multiplier, slice_size = args.multiplier, args.slice
    reference_path = f'{args.base_path}/REFERENCE.csv'
    x, y = (dataset.Loader(args.base_path, reference_path)
            .load_as_x_y_for_ml(normalize=True,
                                augmentation_multiplier=multiplier,
                                augmentation_slice_size=slice_size))
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=42, test_size=0.3)
    if args.type == 'RF':
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif args.type == 'SVM':
        classifier = SVC(random_state=42)
    elif args.type == 'XGBoost':
        classifier = XGBClassifier(objective='multi:softmax', tree_method='gpu_hist', num_class=9, random_state=42)
    elif args.type == 'TPOT':
        classifier = tpot.TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=1)
    else:
        raise Exception(f'Unknown classifier name {args.type}')
    print(f'{datetime.datetime.now()} {args.type} Train started')
    classifier.fit(x_train, y_train)
    print(f'{datetime.datetime.now()} {args.type} Train finished')
    classifier.export('tpot_pipeline.py')
    train_accuracy = test.eval_ml(x_train, y_train, classifier)
    test_accuracy = test.eval_ml(x_test, y_test, classifier)
    print(f'{args.type} Train acc: {train_accuracy}. Test acc: {test_accuracy}')
    save_name = os.path.join(f'{datetime.datetime.now()}_{args.type}', 'model.joblib')
    utils.save_ml(classifier, save_name)


def main():
    parser = argparse.ArgumentParser(description='Training script of ECG problem.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs.')
    parser.add_argument('--batch', type=int, default=1500, help='Batch size.')
    parser.add_argument('--slice', type=int, default=2500, help='Wide of augmentation window.')
    parser.add_argument('--multiplier', type=int, default=40,
                        help='Number of repeats of augmentation process. 0 - disable augmentation')
    parser.add_argument('--print_every', type=int, default=30, help='Print every # iterations.')
    parser.add_argument('--num_classes', type=int, default=9, help='Num classes.')
    parser.add_argument('--type', choices=['CNN', 'MLP', 'VGGLikeCNN',
                                           'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                                           'RF', 'SVM', 'XGBoost', 'TPOT'], default='VGGLikeCNN',
                        help='Type of Classifier or Network')
    parser.add_argument('--base_path', type=str, default='./TrainingSet1', help='Base path to train data directory')
    args = parser.parse_args()

    # draw()
    if args.type in ['CNN', 'MLP', 'VGGLikeCNN',
                     'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19']:
        train(args)
    else:
        train_ml(args)


if __name__ == '__main__':
    main()
