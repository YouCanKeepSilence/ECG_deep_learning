import argparse
import datetime
import os

import tpot
from sklearn.model_selection import RandomizedSearchCV

import utils
import torch
import torch.autograd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier
import numpy as np

import models
import dataset

import sklearn.model_selection

USE_CV_OPTIMUM_RF = True

torch.manual_seed(42)
np.random.seed(42)


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
    train_loader = DataLoader(train_df, batch_size=args.batch, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=args.batch, num_workers=args.num_workers, shuffle=True)
    use_cuda = torch.cuda.is_available()

    if args.type.startswith('VGG_'):
        net = models.get_vgg(args.type.split('_')[-1], batch_norm=True, num_classes=args.num_classes)
    elif args.type.startswith('VGGLikeCNN'):
        pooling = 'max' if len(args.type.split('_')) == 1 else 'avg'
        net = models.VGGLikeCNN(num_classes=args.num_classes, pooling=pooling)
    elif args.type.startswith('CNN_') or args.type == 'CNN':
        pooling = 'max' if len(args.type.split('_')) == 1 else 'avg'
        net = models.CNN(num_classes=args.num_classes, pooling=pooling)
    else:
        net = getattr(models, args.type)(num_classes=args.num_classes)
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
                val_loss, val_acc, _ = test.evaluate(net, test_loader, criterion)
                utils.write_checkpoint(writer, e, args.epochs, i + 1, iteration_per_epochs, acc, loss, val_acc, val_loss)

        val_loss, val_acc, _ = test.evaluate(net, test_loader, criterion)
        label = utils.write_checkpoint(writer, e, args.epochs, iteration_per_epochs,
                                       iteration_per_epochs, acc, loss, val_acc, val_loss)
        checkpoint_name = os.path.join(checkpoint_prefix, f'e_{e}_(step_{label}).pth')
        utils.save_net_model(net, checkpoint_name)
        print(f'Checkpoint of epoch {e} saved'
              f'-----------------------------------------\n'
              )


def train_ml(args):
    multiplier, slice_size = args.multiplier, args.slice
    reference_path = f'{args.base_path}/REFERENCE.csv'
    print(f'{datetime.datetime.now()} Loading data')
    x, y = (dataset.Loader(args.base_path, reference_path)
            .load_as_x_y_for_ml(normalize=True,
                                augmentation_multiplier=multiplier,
                                augmentation_slice_size=slice_size))
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=42, test_size=0.3)
    if args.type == 'RF':
        forest_options = {}
        if USE_CV_OPTIMUM_RF:
            # CV-3 value 0.4083960502053412. Optimum for 100 iteration
            forest_options = {
                'bootstrap': False,
                'criterion': 'gini',
                'max_depth': 45,
                'max_features': 'auto',
                'min_samples_leaf': 1,
                'min_samples_split': 6,
                'n_estimators': 410
            }
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1, **forest_options)
    elif args.type == 'SVM':
        classifier = SVC(random_state=42)
    elif args.type == 'XGBoost':
        classifier = XGBClassifier(objective='multi:softmax', num_class=9, random_state=42)
    elif args.type == 'TPOT':
        classifier = tpot.TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=1)
    else:
        raise Exception(f'Unknown classifier name {args.type}')
    print(f'{datetime.datetime.now()} {args.type} Train started')
    classifier.fit(x_train, y_train)
    print(f'{datetime.datetime.now()} {args.type} Train finished')
    if args.type == 'TPOT':
        classifier.export('tpot_pipeline.py')
    train_accuracy, _ = test.eval_ml(x_train, y_train, classifier)
    test_accuracy, _ = test.eval_ml(x_test, y_test, classifier)
    print(f'{args.type} Train acc: {train_accuracy}. Test acc: {test_accuracy}')
    if args.type != 'TPOT':
        save_name = os.path.join(f'{datetime.datetime.now()}_{args.type}', 'model.joblib')
        utils.save_ml(classifier, save_name)


def tune_ml_params(args):
    multiplier, slice_size = args.multiplier, args.slice
    reference_path = f'{args.base_path}/REFERENCE.csv'
    print(f'{datetime.datetime.now()} Loading data')
    x, y = (dataset.Loader(args.base_path, reference_path)
            .load_as_x_y_for_ml(normalize=True,
                                augmentation_multiplier=multiplier,
                                augmentation_slice_size=slice_size))
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=42, test_size=0.3)
    if args.type == 'RandomizedRF':
        import scipy.stats as st
        params_grid = {
            'n_estimators': [10 ** x for x in range(10, 1011, 50)],
            'max_features': ['auto', 'sqrt'],
            'max_depth': list(range(10, 100)) + [None],
            'min_samples_split': st.randint(2, 10),
            'min_samples_leaf': st.randint(1, 10),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
        estimator = RandomForestClassifier()
        params_fitter = RandomizedSearchCV(estimator=estimator, param_distributions=params_grid,
                                           n_iter=10, cv=3, verbose=2, n_jobs=-1,
                                           random_state=42)
    else:
        raise Exception(f'Unknown classifier name {args.type}')
    start_time = datetime.datetime.now()
    print(f'{start_time} {args.type} Train started')
    params_fitter.fit(x_train, y_train)
    end_time = datetime.datetime.now()
    print(f'{end_time} {args.type} Train finished. Elapsed time {(end_time - start_time).total_seconds()} secs.')
    print(f'Best params: {params_fitter.best_params_}, Best result: {params_fitter.best_score_}')
    train_accuracy, _ = test.eval_ml(x_train, y_train, params_fitter)
    test_accuracy, _ = test.eval_ml(x_test, y_test, params_fitter)
    print(f'{args.type} Train acc: {train_accuracy}. Test acc: {test_accuracy}')
    if args.type != 'TPOT':
        save_name = os.path.join(f'{datetime.datetime.now()}_{args.type}', 'model.joblib')
        utils.save_ml(params_fitter, save_name)


def main():
    parser = argparse.ArgumentParser(description='Training script of ECG problem.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs.')
    parser.add_argument('--batch', type=int, default=1500, help='Batch size.')
    parser.add_argument('--slice', type=int, default=2500, help='Wide of augmentation window.')
    parser.add_argument('--multiplier', type=int, default=10,
                        help='Number of repeats of augmentation process. 0 - disable augmentation')
    parser.add_argument('--print_every', type=int, default=30, help='Print every # iterations.')
    parser.add_argument('--num_classes', type=int, default=9, help='Num classes.')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers to loader.')
    parser.add_argument('--type', choices=['CNN', 'CNN_a', 'MLP', 'VGGLikeCNN', 'VGGLikeCNN_a',
                                           'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                                           'VGG_11a', 'VGG_13a', 'VGG_16a', 'VGG_19a', 'CNNFromArticle',
                                           'RF', 'SVM', 'XGBoost', 'RandomizedRF'], default='CNN',
                        help='Type of Classifier or Network')
    parser.add_argument('--base_path', type=str, default='./TrainingSet1', help='Base path to train data directory')
    args = parser.parse_args()

    print(f'{datetime.datetime.now()} Launched with params: {args}')

    if args.type in ['CNN', 'CNN_a', 'MLP', 'VGGLikeCNN', 'VGGLikeCNN_a',
                     'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                     'VGG_11a', 'VGG_13a', 'VGG_16a', 'VGG_19a', 'CNNFromArticle']:
        train(args)
    elif args.type in ['RF', 'SVM', 'XGBoost', 'TPOT']:
        train_ml(args)
    elif args.type in ['RandomizedRF']:
        tune_ml_params(args)
    else:
        raise Exception(f'Unknown model type {args.type}')


# Fix to use in colab
if __name__ == '__main__':
    import test
    main()
else:
    from . import test
