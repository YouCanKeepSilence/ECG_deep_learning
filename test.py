import argparse
import datetime
import os

import torch
import torchvision
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
    parser.add_argument('--type', choices=['CNN', 'CNN_a', 'MLP', 'VGGLikeCNN', 'VGGLikeCNN_a',
                                           'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                                           'VGG_11a', 'VGG_13a', 'VGG_16a', 'VGG_19a',
                                           'RF', 'SVM', 'XGBoost'], default='CNN_a',
                        help='Type of Classifier or Network')
    parser.add_argument('--base_path', type=str, default='./TrainingSet1', help='Base path to data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers to loader.')
    parser.add_argument('--batch', type=int, default=1, help='Batch size.')
    parser.add_argument('--model_file', type=str,
                        default='CNN_a.pth',
                        help='Name of model weights file relative to ./models folder')
    parser.add_argument('--save_onnx', type=bool, default=False, help='Use to save model as .onnx')
    args = parser.parse_args()
    model = utils.create_model_by_name(args.type, args.model_file)
    reference_path = os.path.join(args.base_path, 'REFERENCE.csv')
    data_loader = dataset.Loader(args.base_path, reference_path)
    if args.type in ['CNN', 'CNN_a', 'MLP', 'VGGLikeCNN', 'VGGLikeCNN_a',
                     'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                     'VGG_11a', 'VGG_13a', 'VGG_16a', 'VGG_19a']:
        if torch.cuda.is_available():
            model = model.cuda()

        print(f'{datetime.datetime.now()} start loading data')
        df = data_loader.load_as_df_for_net(normalize=True)
        df = dataset.ECGDataset(df, 1, random_state=13)
        loader = DataLoader(df, batch_size=args.batch, num_workers=args.num_workers)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'{datetime.datetime.now()} evaluating...')
        val_loss, val_acc = evaluate(model, loader, criterion)
        if args.save_onnx:
            dummy_input_ecg = torch.randn(10, 12, 2500)
            dummy_input_non_ecg = torch.randn(10, 2)
            if torch.cuda.is_available():
                dummy_input_non_ecg, dummy_input_ecg = dummy_input_non_ecg.cuda(), dummy_input_ecg.cuda()
            torch.onnx.export(model, (dummy_input_non_ecg, dummy_input_ecg),
                              f'{args.type}.onnx', verbose=True, input_names=['non_ecg', 'ecg'],
                              output_names=['classes']
                              )
        print(f'{datetime.datetime.now()} {args.type} full dataset accuracy: {val_acc}')
    elif args.type in ['SVM', 'RF', 'XGBoost', 'TPOT']:
        print(f'{datetime.datetime.now()} start loading data')
        x, y = data_loader.load_as_x_y_for_ml(normalize=True)
        print(f'{datetime.datetime.now()} evaluating...')
        acc = eval_ml(x, y, model)
        print(f'{datetime.datetime.now()} {args.type} full dataset accuracy: {acc}')
    else:
        raise Exception(f'Unknown model type {args.type}')


if __name__ == '__main__':
    main()
