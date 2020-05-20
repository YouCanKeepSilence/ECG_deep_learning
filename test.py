import argparse

import torch
from sklearn.metrics import accuracy_score

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


def eval_ml(x_train, x_test, y_train, y_test, classifier):
    y_test_pred = classifier.predict(x_test)
    y_train_pred = classifier.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    return train_accuracy, test_accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluation script of ECG problem.')
    parser.add_argument('--type', choices=['CNN', 'MLP', 'RF', 'SVM', 'XGBoost'], default='CNN',
                        help='Type of Classifier or Network')
    parser.add_argument('--base_path', type=str, default='./TrainingSet1', help='Base path to data directory')
    parser.add_argument('--model_path', type=str, default='./models/CNN.pth', help='Path to model weights file')
    args = parser.parse_args()
    model = utils.create_model_by_name(args.type, args.model_path)
    # TODO load data
    # TODO evaluate model


if __name__ == '__main__':
    main()
