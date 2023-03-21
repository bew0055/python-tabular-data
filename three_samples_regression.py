#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

"""
The purpose of this script is to create linear regressions of three different species on a scatter 
plot using data from the 'iris.csv' file. This can be done by following the tutorial from this 
assignment and breaking it into three seperate sets of data.
"""

def plot_data():
    dataframe = pd.read_csv("iris.csv")
    
    setosa = dataframe[dataframe.species == "Iris_setosa"]
    virginica = dataframe[dataframe.species == "Iris_virginica"]
    versicolor = dataframe[dataframe.species == "Iris_versicolor"]
 
    a = setosa.petal_length_cm
    b = setosa.sepal_length_cm
    regression = stats.linregress(a, b)
    slope = regression.slope
    intercept = regression.intercept
    plt.scatter(a, b, label = 'Setosa Petal v Sepal Lengths (cm)')
    plt.plot(a, slope * a + intercept, color = "red", label = 'Fitted line')
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.legend()
    plt.savefig("Setosa_regress.png")
    plt.clf()
    # need plt.clf in order to clear the prior plot in preparation for the next one - in lieu of quitting the python environment and loading a new one


    c = virginica.petal_length_cm
    d = virginica.sepal_length_cm
    regression = stats.linregress(c, d)
    slope = regression.slope
    intercept = regression.intercept
    plt.scatter(c, d, label = 'Virginica Petal v Sepal Lengths (cm)')
    plt.plot(c, slope * c + intercept, color = "blue", label = 'Fitted line')
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.legend()
    plt.savefig("Virginica_regress.png")
    plt.clf()

    x = versicolor.petal_length_cm
    y = versicolor.sepal_length_cm
    regression = stats.linregress(x, y)
    slope = regression.slope
    intercept = regression.intercept
    plt.scatter(x, y, label = 'Versicolor Petal v Sepal Lengths (cm)')
    plt.plot(x, slope * x + intercept, color = "green", label = 'Fitted line')
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.legend()
    plt.savefig("Versicolor_regress.png")
    plt.clf()

if __name__ == '__main__':
    plot_data()
