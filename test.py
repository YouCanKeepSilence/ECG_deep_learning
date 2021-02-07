import argparse
import datetime
import logging
import os

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np

import dataset
import utils


def calc_score(A):
    F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
    F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
    F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
    F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
    F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
    F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
    F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))

    F1 = (F11 + F12 + F13 + F14 + F15 + F16 + F17 + F18 + F19) / 9

    Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
    Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
    Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))

    logging.info('Final competition scores')
    logging.info('Total File Number: ', np.sum(A))

    logging.info('F11: ', F11)
    logging.info('F12: ', F12)
    logging.info('F13: ', F13)
    logging.info('F14: ', F14)
    logging.info('F15: ', F15)
    logging.info('F16: ', F16)
    logging.info('F17: ', F17)
    logging.info('F18: ', F18)
    logging.info('F19: ', F19)
    logging.info('------')
    logging.info('F1: ', F1)

    logging.info('Faf: ', Faf)
    logging.info('Fblock: ', Fblock)
    logging.info('Fpc: ', Fpc)
    logging.info('Fst: ', Fst)


def evaluate(model, test_loader, criterion, persist_predictions=False):
    model.eval()
    use_cuda = torch.cuda.is_available()
    predictions = []
    with torch.no_grad():
        avg_loss = 0.0
        val_acc = 0
        for non_ecg, ecg, y in test_loader:
            if use_cuda:
                non_ecg, ecg, y = non_ecg.cuda(), ecg.cuda(), y.cuda()
            out = model(non_ecg, ecg)
            loss = criterion(out, y)
            _, pred = torch.max(out.data, 1)
            if persist_predictions:
                predictions.extend(pred.cpu().numpy())
            val_acc += pred.eq(y).sum().item() / y.size(0)
            avg_loss += loss.item()
        avg_loss /= len(test_loader)
        val_acc /= len(test_loader)
    model.train()
    return avg_loss, val_acc, np.array(predictions)


def eval_ml(x, y, classifier):
    y_pred = classifier.predict(x)
    return accuracy_score(y, y_pred), y_pred


def test(args):
    model = utils.create_model_by_name(args.type, args.model_file)
    reference_path = os.path.join(args.base_path, 'REFERENCE.csv')
    data_loader = dataset.Loader(args.base_path, reference_path)

    if args.type in ['CNN', 'CNN_a', 'MLP', 'VGGLikeCNN', 'VGGLikeCNN_a',
                     'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                     'VGG_11a', 'VGG_13a', 'VGG_16a', 'VGG_19a']:
        if torch.cuda.is_available():
            model = model.cuda()

        logging.info(f'Start loading data')
        df = data_loader.load_as_df_for_net(normalize=True, save_df=True)
        df = dataset.ECGDataset(df, slices_count=1, random_state=13, slice_len=2500)
        loader = DataLoader(df, batch_size=args.batch, num_workers=args.num_workers, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()

        start_time = datetime.datetime.now()
        logging.info(f'{start_time} evaluating...')
        val_loss, val_acc, predictions = evaluate(model, loader, criterion, persist_predictions=args.calc_score)
        end_time = datetime.datetime.now()
        delta = (end_time - start_time).total_seconds()
        logging.info(f'{end_time} {args.type} Inference time: {delta} secs. '
              f'(Per record: {delta * 1000 / len(df)} msecs / record) \n'
              f'Full dataset accuracy: {val_acc} \n'
              f'Records count: {len(df)} \n')

        if args.calc_score:
            true_y = []
            for _, _, y in loader:
                true_y.extend(y.numpy())
            true_y = np.array(true_y)
            conf = confusion_matrix(true_y, predictions)
            calc_score(conf)
            np.save('./confusion_matrix', conf)

        if args.save_onnx:
            logging.info(f'Save model to .onnx')
            dummy_input_ecg = torch.randn(10, 12, 2500)
            dummy_input_non_ecg = torch.randn(10, 2)
            if torch.cuda.is_available():
                dummy_input_non_ecg, dummy_input_ecg = dummy_input_non_ecg.cuda(), dummy_input_ecg.cuda()

            torch.onnx.export(
                model, (dummy_input_non_ecg, dummy_input_ecg), f'{args.type}.onnx',
                verbose=True, input_names=['non_ecg', 'ecg'], output_names=['classes']
            )

            logging.info(f'.onnx saved')

    elif args.type in ['SVM', 'RF', 'XGBoost', 'TPOT']:
        logging.info(f'Start loading data')
        x, y = data_loader.load_as_x_y_for_ml(normalize=True)

        start_time = datetime.datetime.now()
        logging.info(f'{start_time} evaluating...')
        acc, predictions = eval_ml(x, y, model)
        end_time = datetime.datetime.now()
        delta = (end_time - start_time).total_seconds()
        logging.info(
            f'{end_time} {args.type} Inference time: {delta} secs. '
            f'(Per record: {delta * 1000 / y.shape[0]} msecs / record) \n'
            f'Full dataset accuracy: {acc} \n'
            f'Records count: {y.shape[0]} \n'
        )

        if args.calc_score:
            conf = confusion_matrix(y, predictions)
            logging.info(conf)
            calc_score(conf)

    else:
        raise Exception(f'Unknown model type {args.type}')


def main():
    parser = argparse.ArgumentParser(description='Evaluation script of ECG problem.')
    parser.add_argument('--type', choices=['CNN', 'CNN_a', 'MLP', 'VGGLikeCNN', 'VGGLikeCNN_a',
                                           'VGG_11', 'VGG_13', 'VGG_16', 'VGG_19',
                                           'VGG_11a', 'VGG_13a', 'VGG_16a', 'VGG_19a',
                                           'RF', 'SVM', 'XGBoost'], default='CNN_a',
                        help='Type of Classifier or Network')
    parser.add_argument('--base_path', type=str, default='./TrainingSet1', help='Base path to data directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Num workers to loader.')
    parser.add_argument('--batch', type=int, default=1, help='Batch size.')
    parser.add_argument('--model_file', type=str,
                        default='CNN_a/CNN_a_with_preprocessing.pth',
                        help='Name of model weights file relative to ./models folder')
    parser.add_argument('--save_onnx', type=bool, default=False, help='Use to save model as .onnx')
    parser.add_argument('--calc_score', type=bool, default=True, help='Use to calculate competition score')

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    utils.init_logger()
    main()
