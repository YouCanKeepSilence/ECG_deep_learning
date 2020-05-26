# ECG illness predict problem using Deep Neural Networks
## Short description
Automatic identification of the rhythm/morphology abnormalities in 12-lead ECGs. 
Original competition [link](http://2018.icbeb.org/Challenge.html)

Dataset contains age, gender and 12-lead ECG data of patients with or without illness.
 
Label field classify all patients to 9 classes:

| Label  | Description  | 
| :---: | :--- |
| 1 | Normal |
| 2 | Atrial fibrillation (AF) |
| 3 | First-degree atrioventricular block (I-AVB) |
| 4 | Left bundle branch block (LBBB) |
| 5 | Right bundle branch block (RBBB) |
| 6 | Premature atrial contraction (PAC) |
| 7 | Premature ventricular contraction (PVC) |
| 8 | ST-segment depression (STD) |
| 9 | ST-segment elevated (STE) |


Target - predict illness "label" by rest columns from dataset.


## Installation
To setup all dependencies just use:

`pip install -r requirements.txt`

## Train
To train you can download dataset from [here](https://drive.google.com/open?id=1Et6O5ihcFuPDXgnkTnUvuwTao0Pmayvq)

And start training by 

`python train.py --base_path=${YOUR_DATA_FOLDER_PATH}`

## Test
To test you can download needed model file from [here](https://drive.google.com/open?id=1aIyH4n2bxR1vX3d95IOmp5dsQOIeh9U2)

`python test.py --base_path=${YOUR_DATA_FOLDER_PATH} --type=${NEEDED_MODEL_TYPE} --model_path=${PATH_TO_MODEL_FILE}`

## Model scores
Networks were tested with 2500 slice size and 40 augmentation multiplier. 
Test size was 30% of augmented dataset.

All networks learn for 20 epochs. After that I tried to get 
best checkpoint and write result of them.

To measure full dataset accuracy were generated 1 slice with 2500 
length with another random_state and batch_size = 1

You can find TensorBoard files of NN learning process and model weights files [here](https://drive.google.com/open?id=1aIyH4n2bxR1vX3d95IOmp5dsQOIeh9U2) 

SVM, RF, XGBoost were tested without augmentation. 

_a suffix mean that network uses avg pooling instead of max.

| Model  | Train accuracy  |  Test accuracy  | Full dataset accuracy  |
|:---:|:---:|:---:|:---:|
|CNN   | 95% | 65% | 84% |
|CNN_a   |  95% | 65%  |  88% |
|VGGLikeCNN   | 96%  | 66%  | 87%  |
|VGGLikeCNN_a   | 97%  | 66%  | 88%  |
|<b>VGG_11</b>   | <b>94%</b>  | <b>73%</b>  | <b>90% </b> |
|MLP   | 66%  | 56%  | 63%  |
|SVM   | 73%   | 49%  | 66%  |
|Random Forest | 98%  | 51% | 84%  |
|XGBoost   | 98%  | 57%  | 86%  |


