import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
"""
Data frame format:
    - Columns:
        - 'id'
        - 'Entity'
        - 'Code'
        - 'Year'
        - 'Life expectancy
"""


def reshape_data(df, entity='Sweden'):
    """
    extracts the data for the given entity and reshapes it to a numpy array.
    """
    # select data for the given entity
    df = df.loc[df['Entity'] == entity]

    x = np.array(df['Year'])
    # convert every x value to a list with one element
    x = x.reshape(-1, 1)

    y = np.array(df['Life expectancy'])
    # convert every y value to a list with one element
    y = y.reshape(-1, 1)

    return x, y


def poly_regression(x, y, year_to_predict=2100, entity='Sweden', degree=2):
    """
    performs polynomial regression on the given data and predicts the future and then plots everything.
    """
    # create polynomial features
    poly_regression = PolynomialFeatures(degree=degree)
    x_poly = poly_regression.fit_transform(x)
    # create linear regression model
    lin_regression = LinearRegression()
    # fit the model
    lin_regression.fit(x_poly, y)

    #create a list of years to predict up to the given year to predict
    years = np.arange(x[-1][0], year_to_predict)
    # create a list of predicted values
    predicted_values = lin_regression.predict(poly_regression.fit_transform(years.reshape(-1, 1)))


    # plot the data and the predicted line from the first year to the given year
    plt.plot(years, predicted_values, color='red', label='Predicted')
    plt.plot(x, y, color='black', label='Actual Data')
    plt.plot(x, lin_regression.predict(x_poly), color='blue', label='Regression')
    plt.title('Life expectancy of {}'.format(entity))
    plt.xlabel('Year')
    plt.ylabel('Life expectancy')
    plt.legend()
    plt.show()

def main():
    # adjustable parameters
    endity = 'United States' # Country to predict
    year_to_predict = 2050 # draw a line to predict up to this year
    degree = 1 # degree of the polynomial

    # read data
    df = pd.read_csv('life_expectancy.csv')
    # split data
    x, y = reshape_data(df, entity=endity)
    # perform polynomial regression
    poly_regression(x, y, year_to_predict=year_to_predict, entity=endity, degree=degree)

if __name__ == "__main__":
    main()