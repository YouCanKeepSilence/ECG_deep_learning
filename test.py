import argparse
import os

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import dataset
import utils


def evaluate(model, test_loader, criterion):
    model.eval()
    use_cuda = torch.cuda.is_available()
    with torch.no_grad():
        avg_loss = 0.0
        val_acc = 0
        for non_ecg, ecg, y in test_loader:
            if use_cuda:
                non_ecg, ecg, y = non_ecg.cuda(), ecg.cuda(), y.cuda()
            out = model(non_ecg, ecg)
            loss = criterion(out, y)
            _, pred = torch.max(out.data, 1)
            val_acc += pred.eq(y).sum().item() / y.size(0)
            avg_loss += loss.item()
        avg_loss /= len(test_loader)
        val_acc /= len(test_loader)
    model.train()
    return avg_loss, val_acc


def eval_ml(x, y, classifier):
    y_pred = classifier.predict(x)
    return accuracy_score(y, y_pred)


def main():
    parser = argparse.ArgumentParser(description='Evaluation script of ECG problem.')
    parser.add_argument('--type', choices=['CNN', 'MLP', 'RF', 'SVM', 'XGBoost'], default='XGBoost',
                        help='Type of Classifier or Network')
    parser.add_argument('--base_path', type=str, default='./TrainingSet1', help='Base path to data directory')
    parser.add_argument('--model_file', type=str, default='2020-05-21 20:10:23.962904_XGBClassifier/model.joblib', help='Name of model weights file')
    args = parser.parse_args()
    model = utils.create_model_by_name(args.type, args.model_file)
    reference_path = os.path.join(args.base_path, 'REFERENCE.csv')
    data_loader = dataset.Loader(args.base_path, reference_path)
    if args.type in ['CNN', 'MLP']:
        if torch.cuda.is_available():
            model = model.cuda()

        df = data_loader.load_as_df_for_net(normalize=True)
        df = dataset.ECGDataset(df, 1, random_state=13)
        loader = DataLoader(df, batch_size=1, num_workers=4)
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate(model, loader, criterion)
        print(f'{args.type} full dataset accuracy: {val_acc}')
    elif args.type in ['SVM', 'RF', 'XGBoost']:
        x, y = data_loader.load_as_x_y_for_ml(normalize=True)
        acc = eval_ml(x, y, model)
        print(f'{args.type} full dataset accuracy: {acc}')
    else:
        raise Exception(f'Unknown model type {args.type}')


if __name__ == '__main__':
    main()
