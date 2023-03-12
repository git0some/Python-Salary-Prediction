import os
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)


# read data, create df
def load_data():
    data = pd.read_csv('../Data/data.csv')
    return data


def model_fit(data, use_col):
    X = data.drop(use_col, axis=1)
    X.drop('salary', inplace=True, axis=1)
    y = data['salary']
    if len(X.shape) == 1:
        X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = train_model(X_train, y_train)
    return model


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def mape_handling_negatives(model, X_test, y_test, y_train):
    # predict salary from X_test
    predicted_salary = pd.DataFrame(model.predict(X_test))
    # replace negative predicted salary with zero
    predicted_salary_with_negatives_as_zeros = predicted_salary.where(lambda x: x > 0, 0)
    # replace negative predicted salary with median from y_train
    predicted_salary_with_negatives_as_median = predicted_salary.where(lambda x: x > 0, y_train.median())
    mape_results = [mape(y_test, predicted_salary_with_negatives_as_zeros),
                    mape(y_test, predicted_salary_with_negatives_as_median)]
    return min(mape_results)


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['age', 'experience', "salary"], axis=1),
                                                        data["salary"], test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mape_results = mape_handling_negatives(model, X_test, y_test, y_train)
    print(round(mape_results, 5))


if __name__ == '__main__':
    main()
