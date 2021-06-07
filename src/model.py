
import pandas as pd
import plotnine as p9

from sklearn.model_selection import train_test_split
from sklearn import ensemble

bookings = pd.read_csv('data/freenow_bookings.csv',
                       warn_bad_lines=False, error_bad_lines=False)

def tranform_data(bookings):
  """Transform booking information into data useful for modelling taxi fares
  """

  # -- parse dates and compute duration
  bookings['date_close_passenger'] = pd.to_datetime(bookings['date_close_passenger'])
  bookings['carry_date'] = pd.to_datetime(bookings['carry_date'])

  # -- unsure this is reliable

  bookings['duration'] = bookings['date_close_passenger'] - bookings['carry_date']
  # -- essential columns must be present; filter for rows with all of them
  df = bookings[bookings[['route_distance',
                          'tour_value', 'duration']].notnull().all(1)]

  df = df[df['route_distance'] < 10000][df['duration'] < pd.Timedelta(1, 'h')]
  df["duration"] = (df["duration"]).dt.seconds

  df = df.query("state!='CANCELED'")

  return df

def model_gradient_boosting(df):
  """Model lower, mid, and upper bounds for taxi-fares"""

  LOWER_ALPHA = 0.1
  UPPER_ALPHA = 0.9

  labels = df['tour_value'].reset_index(drop=True)
  train0 = df[["duration", "route_distance"]].reset_index(drop=True)
  x_train, x_test, y_train, y_test = train_test_split(train0, labels, random_state=123)

  lower_model = ensemble.GradientBoostingRegressor(
    loss="quantile", alpha=LOWER_ALPHA)
  lower_model.fit(x_train, y_train)

  mid_model = ensemble.GradientBoostingRegressor(
      loss="ls")
  mid_model.fit(x_train, y_train)

  upper_model = ensemble.GradientBoostingRegressor(
      loss="quantile", alpha=UPPER_ALPHA)
  upper_model.fit(x_train, y_train)

  predictions = pd.DataFrame(y_test)

  predictions['idx'] = predictions.index
  predictions['lower'] = lower_model.predict(x_test)
  predictions['mid'] = mid_model.predict(x_test)
  predictions['upper'] = upper_model.predict(x_test)

  predictions.sort_values(by=['tour_value'])

  predictions['mislabelled'] = (predictions['tour_value'] > predictions['upper']) | (predictions['tour_value'] < predictions['lower'])
  predictions['mislabelled'] = predictions['mislabelled'].map({True: 'Yes', False: 'No'})

  plot = (
    p9.ggplot(predictions) +
    p9.geom_histogram(p9.aes(x='mislabelled', fill='mislabelled'))
  )

  plot.save('model.png')

df = tranform_data(bookings)

model_gradient_boosting(df)
