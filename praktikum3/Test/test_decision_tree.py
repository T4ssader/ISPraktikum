import numpy
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ..DecisionTree import DecisionTreeClassifier
from .. import Visualisation
from .. import node
import os

'''
Initialize test Data
'''
# IRIS DATA
col_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
diabetes_data = pd.read_csv("praktikum3/datasets/iris.csv", skiprows=1, header=None, names=col_names)

diabetes_X = diabetes_data.iloc[:, :-1].values
diabetes_Y = diabetes_data.iloc[:, -1].values.reshape(-1, 1)

diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=.2, random_state=41)



# DIABETES DATA
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_data = pd.read_csv("praktikum3/datasets/diabetes.csv", skiprows=1, header=None, names=col_names)

diabetes_X = diabetes_data.iloc[:, :-1].values
diabetes_Y = diabetes_data.iloc[:, -1].values.reshape(-1, 1)

diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=.2, random_state=41)



'''
Tests
'''
def test_iris_tree():
    iris_dt = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    iris_dt.fit(diabetes_X_train, diabetes_Y_train)

    iris_viz = Visualisation.Visualisation(iris_dt, diabetes_data)

    iris_viz.save()

    Y_pred = iris_dt.predict(diabetes_X_test)

    assert 0.9 <=  accuracy_score(diabetes_Y_test, Y_pred)

def test_diabetes_tree():
    diabetes_dt = DecisionTreeClassifier(min_samples_split=600, max_depth=3)
    diabetes_dt.fit(diabetes_X_train, diabetes_Y_train)

    diabetes_viz = Visualisation.Visualisation(diabetes_dt, diabetes_data)

    diabetes_viz.save()

    Y_pred = diabetes_dt.predict(diabetes_X_test)

    assert 0.9 <=  accuracy_score(diabetes_Y_test, Y_pred)


'''TODO

overfitting ( baum mit einer node fast so accurate wie max baum mit depth = numpy.infinity)
nur glucose 600 split
'''