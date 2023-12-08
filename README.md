## Toxicity prediction model
This multilabel classification task was completed as an assignment in Machine Learning course and a [kaggle competition](https://www.kaggle.com/competitions/kmaml223), using the dataset provided in the competition.
The model estimates probabilities of sentence belonging to any of 6 categories:
```
[toxic, severe_toxic, obscene	threat, insult, identity_hate]
```

<br>

### Usage
To get prediction the `prediction_script.py` script is provided*. It takes prompt as command line argument, e.g.:
```
python prediction_script.py I am very toxic today!☣️☣️☣️
```
The model recognizes English words, numbers and emojies. As an output, it generates probabilities for each of the mentioned classes:
| toxic | severe_toxic | obscene | threat | insult | identity_hate |
|-------|--------------|---------|--------|--------|---------------|
| 0.321 | 0.000        | 0.013   | 0.001  | 0.019  | 0.000         |


Alternatively, included Dockerfile can be used to run `prediction_script.py` inside container:


```
docker build -t toxicity .
docker run toxicity I am very toxic today!☣️☣️☣️
```
<br>
*Unfortunantely it was not possible to upload tokenizer to github, so it has to be generated manually by performing sequence Preprocessing-1, Preprocessing-2, Submission. It may be a lengthy process. I am working on solving this issue.

### Model
The model used in this solution is XGBoost Booster model. The trained model has following performance:

|Accuracy|	Precision|	Recall|	F1	|AUC|
|--------|---------|------|----|---|
|0	|0.917249	|0.854725	|0.539923	|0.659374	|0.969175|
The weights are stored under `models/xgb-v<latest_version>.json`, and can be loaded using `xgb.Booster().load_model(<filepath>)`.
