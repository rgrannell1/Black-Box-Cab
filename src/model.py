
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble

bookings = pd.read_csv('data/freenow_bookings.csv',
                       warn_bad_lines=False, error_bad_lines=False)

def as_point_obj(point):
  """Extract points as a pair of lng, lat coordinates
  """
  lng, lat = point.replace('POINT(', '').replace(')', '').split(' ')
  return (float(lng), float(lat))

def tranform_data(bookings):
  """Transform booking information into data useful for modelling taxi fares
  """

  # -- parse dates and compute duration
  bookings['date_close'] = pd.to_datetime(bookings['date_close'])
  bookings['carry_date'] = pd.to_datetime(bookings['carry_date'])

  bookings['duration'] = bookings['date_close'] - bookings['carry_date']

  # -- essential columns must be present; filter for rows with all of them
  df = bookings[bookings[['route_distance',
                          'route_price', 'duration']].notnull().all(1)]
  return df[df['route_distance'] < 10000][df['duration'] < pd.Timedelta(1, 'h')]

df = tranform_data(bookings)

def model_1d_linear_regression(df):
  labels = df['tour_value'].reset_index(drop=True)
  train0 = df[["duration"]].reset_index(drop=True)

  x_train, x_test, y_train, y_test = train_test_split(
      train0, labels,  random_state=1)

  polyreg = LinearRegression()
  polyreg.fit(x_train, y_train)

  print("++ duration,distance model fit ++")
  print(polyreg.score(x_test, y_test))

def model_2d_linear_regression(df):
  labels = df['tour_value'].reset_index(drop=True)
  train0 = df[["duration", "route_distance"]].reset_index(drop=True)

  x_train, x_test, y_train, y_test = train_test_split(
      train0, labels, random_state=1)

  polyreg = LinearRegression()

  polyreg.fit(x_train, y_train)

  print("++ duration,distance model fit ++")
  print(polyreg.score(x_test, y_test))

def model_gradient_boosting(df):
  labels = df['tour_value'].reset_index(drop=True)
  train0 = df[["duration", "route_distance"]].reset_index(drop=True)
  x_train, x_test, y_train, y_test = train_test_split(train0, labels,  random_state=1)

  clf = ensemble.GradientBoostingRegressor(learning_rate=0.01)
  clf.fit(x_train, y_train)
  print("++ duration,distance model fit with GBR ++")
  print(clf.score(x_test, y_test))

df["duration"] = (df["duration"]).dt.seconds

model_1d_linear_regression(df)
model_2d_linear_regression(df)
model_gradient_boosting(df)
