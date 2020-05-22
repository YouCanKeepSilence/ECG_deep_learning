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
To train you can download dataset from [here]()

And start training by 

`python train.py --base_path=${YOUR_DATA_FOLDER_PATH}`

## Test
To test you can download needed model file from [here]()

`python test.py --base_path=${YOUR_DATA_FOLDER_PATH} --type=${NEEDED_MODEL_TYPE} --model_path=${PATH_TO_MODEL_FILE}`

## Model scores
Networks were tested with 2500 slice size and 40 augmentation multiplier. 
Test size was 30% of augmentated dataset.

To measure full dataset accuracy were generated 1 slice with 2500 length with another random_state


SVM, RF, XGBoost were tested without augmentation

| Model  | Test accuracy  |  Train accuracy  | Full dataset accuracy  |
|:---:|:---:|:---:|:---:|
|CNN   |  67% | 96%  | 85%  |
|MLP   | 54%  | 76%  | 66%  |
|SVM   | 49%   | 73%  |   |
|Random Forest | 51%  | 98% | 84%  |
|XGBoost   | 57%  | 98%  | 86%  |


