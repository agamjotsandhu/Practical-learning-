import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data = pd.DataFrame(yf.download('AAPL', start = "2025-01-01", end = "2025-02-04"))

dates = data.index
dates_cleaned = []
data.to_csv("historical_data.csv")
prices = data[data.columns[0]]
prices_cleaned = []

for date in dates:
    dates_cleaned.append(int((str(date).split("-"))[2].split()[0]))

for price in prices:
    prices_cleaned.append(float(price))

print(dates_cleaned)
print(prices_cleaned)

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel = "linear", C = 1000)
    svr_poly = SVR(kernel = 'poly', C = 1000, degree = 2)
    svr_rbf = SVR(kernel = 'rbf', C = 1000, gamma = 0.1)

    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model') 
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model') 
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend() 
    plt.show()
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

predicted_price = predict_prices(dates_cleaned, prices_cleaned, 29)