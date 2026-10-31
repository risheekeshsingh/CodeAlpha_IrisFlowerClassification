# CodeAlpha_IrisFlowerClassification

**CodeAlpha Data Science Internship | Task 1: Iris Flower Classification**

A machine learning project that identifies the species of an Iris flower (*setosa*, *versicolor* or *virginica*) from four measurements, built with Python and scikit-learn.

## Result at a glance

| Item | Value |
|---|---|
| Best model | SVM (RBF kernel) |
| Cross-validated accuracy | 96.7% |
| Test accuracy (unseen data) | **96.7%** (29 of 30 flowers correct) |

## Dataset

- 150 flowers, 50 per species
- Features: sepal length, sepal width, petal length, petal width (all in cm)
- Target: species
- Source: the built-in dataset in `sklearn.datasets.load_iris`, also saved as `data/iris.csv`

## Project structure

```
CodeAlpha_IrisFlowerClassification/
├── iris_classification.ipynb   # full notebook with outputs and charts
├── iris_classification.py      # same code as a plain script
├── data/iris.csv               # dataset
├── images/                     # saved charts
├── requirements.txt            # libraries needed
└── README.md
```

## Method

1. **Data quality check:** no missing values, one harmless duplicate row, and 4 mild outliers in sepal width (kept, because they are real flowers).
2. **Exploration:** pair plot and correlation heatmap.
3. **Feature scoring:** ANOVA F-test shows petal length and petal width separate the species best.
4. **Train/test split:** 80% train, 20% test, stratified so each species keeps its share.
5. **Model comparison:** 5-fold cross-validation on the training data only. Scaling is inside a `Pipeline`, so there is no data leakage.
6. **Final evaluation:** the best model is tested once on the held-out test set.

## Model comparison

| Model | CV accuracy |
|---|---|
| SVM (RBF) | 0.967 |
| Logistic Regression | 0.958 |
| KNN (k=5) | 0.958 |
| Decision Tree | 0.950 |
| Random Forest | 0.950 |

## Test set report (SVM)

| Species | Precision | Recall | F1 |
|---|---|---|---|
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 1.00 | 0.90 | 0.95 |
| virginica | 0.91 | 1.00 | 0.95 |

The only mistake was one versicolor flower predicted as virginica. The test set has just 30 flowers, so one error moves accuracy by about 3 points.

## Charts

![Pair plot](images/pairplot.png)
![Confusion matrix](images/confusion_matrix.png)

## Key learnings

- Petal measurements are the strongest predictors of species.
- Setosa is easy to separate; versicolor and virginica overlap slightly.
- Putting preprocessing inside a pipeline prevents information from the test set leaking into training.

## How to run

```bash
git clone https://github.com/risheekeshsingh/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
pip install -r requirements.txt
jupyter notebook iris_classification.ipynb
```




# CodeAlpha_CarPricePrediction

**CodeAlpha Data Science Internship | Task 3: Car Price Prediction with Machine Learning**

A regression project that predicts the price of a used car from its mileage, engine size, make, model, type and extras, built with Python, pandas and scikit-learn.

## Result at a glance

| Item | Value |
|---|---|
| Best model | Gradient Boosting Regressor |
| Test R² | **0.977** |
| Test MAE (average error) | about $987 |
| Test RMSE | about $1,355 |

## Dataset

- `data/car_worth.csv`: 804 used 2005 GM cars
- Target: **Price** (US dollars)
- Features: Mileage, Make, Model, Type, Cylinder, Liter (engine size), Doors, Cruise, Sound, Leather (the last three are 0/1 extras). `Trim` was dropped because it overlaps with Model and Type.
- Source: public dataset from the Wesleyan QAC applied-data course (prices follow the Kelley Blue Book): https://raw.githubusercontent.com/nazzstat/AppliedData/master/PracticeData/car_worth.csv
- Note: this is a public alternative to the dataset linked in the internship brief, which was not accessible when this was built. It has no horsepower column.

## Project structure

```
CodeAlpha_CarPricePrediction/
├── car_price_prediction.ipynb   # full notebook with outputs and charts
├── car_price_prediction.py      # same code as a plain script
├── data/car_worth.csv           # dataset
├── images/                      # saved charts
├── requirements.txt
└── README.md
```

## Method

1. **Data quality:** no missing values and no duplicates. 17 price outliers by the IQR rule; 15 are Cadillacs. They are genuine luxury cars, so they were **kept**, not removed.
2. **Exploration:** price distribution, price by make, correlation heatmap. Cylinder (0.57), Liter (0.56) and Cruise (0.43) correlate most with price; Mileage has a weak negative link (-0.14).
3. **Feature selection:** F-test scores (`f_regression`) for the numeric features.
4. **Preprocessing in a `Pipeline`:** median imputation and scaling for numbers; most-frequent imputation and one-hot encoding for Make, Model and Type. Fitted on training data only, so there is no data leakage.
5. **Model comparison:** 5-fold cross-validation on the training data.
6. **Final test:** the best model is evaluated once on a held-out 20% test set, plus residual plots and permutation importance.

## Model comparison

| Model | CV R² | CV MAE ($) |
|---|---|---|
| Gradient Boosting | 0.978 | 1,115 |
| Linear Regression | 0.978 | 1,127 |
| Random Forest | 0.976 | 1,109 |
| Ridge | 0.976 | 1,181 |

The four models are very close. Plain Linear Regression scores the same as Gradient Boosting, so a simple, easy-to-explain model is already a strong choice here.

## What drives the price?

Permutation importance (drop in R² when a feature is shuffled):

| Feature | Importance |
|---|---|
| Make | 0.805 |
| Liter (engine size) | 0.354 |
| Model | 0.063 |
| Mileage | 0.059 |
| Type | 0.037 |
| Cylinder | 0.035 |

Cruise, Leather, Sound and Doors add almost nothing once make and engine size are known.

## Charts

![Price EDA](images/price_eda.png)
![Predictions](images/predictions.png)
![Feature importance](images/importance.png)

## Key learnings

- Brand (Make) and engine size explain most of a car's price.
- Outliers deserve investigation before removal; here they were real luxury cars.
- Putting preprocessing inside a pipeline prevents leakage, and cross-validation gives a fairer comparison than one split.

## Limitations

The data covers only 2005 GM cars, so predictions will not transfer to other brands or years. It has no horsepower information.

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
pip install -r requirements.txt
jupyter notebook car_price_prediction.ipynb
```

## Author

RISHEEKESH SINGH | [github.com/risheekeshsingh](https://github.com/risheekeshsingh)

Built during the CodeAlpha Data Science Internship (@CodeAlpha).
















## Author

[Your Name] | [LinkedIn profile link] | [github.com/risheekeshsingh](https://github.com/risheekeshsingh)

Built during the CodeAlpha Data Science Internship (@CodeAlpha).
