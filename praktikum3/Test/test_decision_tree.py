import numpy
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ..DecisionTree import DecisionTreeClassifier
from .. import Visualisation
from .. import node
import matplotlib.pyplot as plt

'''
Initialize test Data
'''
# IRIS DATA
iris_col_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
iris_data = pd.read_csv("praktikum3/datasets/iris.csv", skiprows=1, header=None, names=iris_col_names)

iris_X = iris_data.iloc[:, :-1].values
iris_Y = iris_data.iloc[:, -1].values.reshape(-1, 1)

iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = train_test_split(iris_X, iris_Y,
                                                                        test_size=.2, random_state=41)

# DIABETES DATA
diabetes_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                      'DiabetesPedigreeFunction',
                      'Age', 'Outcome']
diabetes_data = pd.read_csv("praktikum3/datasets/diabetes.csv", skiprows=1, header=None, names=diabetes_col_names)

diabetes_X = diabetes_data.iloc[:, :-1].values
diabetes_Y = diabetes_data.iloc[:, -1].values.reshape(-1, 1)

diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y,
                                                                                        test_size=.2, random_state=41)

# HEALTHCARE DATA
healthcare_col_names = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
healthcare_data = pd.read_csv("praktikum3/datasets/healthcare.csv", skiprows=1, header=None, names=healthcare_col_names)

healthcare_X = healthcare_data.iloc[:, :-1].values
healthcare_Y = healthcare_data.iloc[:, -1].values.reshape(-1, 1)

healthcare_X_train, healthcare_X_test, healthcare_Y_train, healthcare_Y_test = train_test_split(healthcare_X,
                                                                                                healthcare_Y,
                                                                                                test_size=.2,
                                                                                                random_state=41)

'''
Tests
'''


def test_iris_tree():
    iris_dt = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    iris_dt.fit(iris_X_train, iris_Y_train)

    iris_viz = Visualisation.Visualisation(iris_dt, iris_data)

    iris_viz.save()

    Y_pred = iris_dt.predict(iris_X_test)

    iris_dt.print_tree()
    assert 0.99 <= accuracy_score(iris_Y_test, Y_pred)


def test_diabetes_tree():
    diabetes_dt = DecisionTreeClassifier(min_samples_split=3, max_depth=2)
    diabetes_dt.fit(diabetes_X_train, diabetes_Y_train)

    diabetes_viz = Visualisation.Visualisation(diabetes_dt, diabetes_data)

    diabetes_viz.save()

    Y_pred = diabetes_dt.predict(diabetes_X_test)

    assert 0.9 <= accuracy_score(diabetes_Y_test, Y_pred)


def test_healthcare_stroke_tree():
    healthcare_dt = DecisionTreeClassifier(min_samples_split=135, max_depth=4)
    healthcare_dt.fit(healthcare_X_train, healthcare_Y_train)

    healthcare_viz = Visualisation.Visualisation(healthcare_dt, healthcare_data)

    healthcare_viz.save()

    Y_pred = healthcare_dt.predict(healthcare_X_test)

    assert 0.99 <= accuracy_score(healthcare_Y_test, Y_pred)


def test_plot_healthcare():


    accuracy_scores = []
    min_sample_split = [2, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 300]
    depths = []
    for depth in range(16):
        # iris_dt = DecisionTreeClassifier(min_samples_split=split, max_depth=3)
        # iris_dt.fit(iris_X_train, iris_Y_train)
        #
        # Y_pred = iris_dt.predict(iris_X_test)
        # accuracy_scores.append(accuracy_score(iris_Y_test, Y_pred))
        # #depths.append(depth)

        # diabetes_dt = DecisionTreeClassifier(min_samples_split=split, max_depth=3)
        # diabetes_dt.fit(diabetes_X_train, diabetes_Y_train)
        #
        # Y_pred = diabetes_dt.predict(diabetes_X_test)
        # accuracy_scores.append(accuracy_score(diabetes_Y_test, Y_pred))
        # depths.append(depth)

        healthcare_dt = DecisionTreeClassifier(min_samples_split=5, max_depth=depth)
        healthcare_dt.fit(healthcare_X_train, healthcare_Y_train)

        Y_pred = healthcare_dt.predict(healthcare_X_test)
        accuracy_scores.append(accuracy_score(healthcare_Y_test, Y_pred))
        depths.append(depth)

    plt.plot(depths, accuracy_scores)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()


'''TODO

overfitting ( baum mit einer node fast so accurate wie max baum mit depth = numpy.infinity)
nur glucose 600 split
'''
