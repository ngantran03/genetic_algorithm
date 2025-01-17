import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import random
from sklearn.model_selection import RandomizedSearchCV

# def function_evaluation(x):
#     # Bukin function N.6
#     # f(x) = 100 * sqrt(abs(x2 - 0.01 * x1^2)) + 0.01 * abs(x1 + 10)
#     fx = 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)
#     return fx

data = pd.read_csv('car_evaluation.csv', header = None)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data.columns = col_names

label_encoder = preprocessing.LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Split the data into features and target variable
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42)


# x = [n_estimators,max_features, max_depth]
def function_evaluation(x):
    # print(x)
    if x[0] == 0 or x[1] == 0 or x[2] == 0:
        INVALID_INPUT = -9999
        return INVALID_INPUT
    else:
        # Train the random forest model
        clf = RandomForestClassifier(n_estimators = abs(x[0]), max_features = abs(x[1]/100), max_depth = abs(x[2]), random_state=42)
        clf.fit(X_train, y_train)

        # Predict the target variable for the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        # print(f'Accuracy: {accuracy}')
        return accuracy
    

def grid_search(domain):
    param_dist = {
        'n_estimators': [random.randint(domain[0][0][0], domain[0][0][1]) for _ in range(10)],
        'max_depth': [random.randint(domain[1][0][0], domain[1][0][1]) for _ in range(10)],
        'min_samples_split': [random.randint(domain[2][0][0], domain[2][0][1]) for _ in range(10)],
    }
        # Randomized search
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=10,  # Limit to 10 evaluations
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Results
    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    # Test set evaluation
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)
