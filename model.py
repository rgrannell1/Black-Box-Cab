
import pandas as pd
import plotnine as p9

bookings = pd.read_csv('data/freenow_bookings.csv', warn_bad_lines=False, error_bad_lines=False)

def as_point_obj (point):
  """Extract points as a pair of lng, lat coordinates
  """
  lng, lat = point.replace('POINT(', '').replace(')','').split(' ')
  return (float(lng), float(lat))

def tranform_data(bookings):
  """Transform booking information into data useful for modelling taxi fares
  """

  # -- parse dates and compute duration
  bookings['date_close'] = pd.to_datetime(bookings['date_close'])
  bookings['carry_date'] = pd.to_datetime(bookings['carry_date'])
  bookings['duration'] = bookings['date_close'] - bookings['carry_date']

  # -- essential columns must be present; filter for rows with all of them
  df = bookings[bookings[['route_distance', 'tour_value', 'duration']].notnull().all(1)]
  return df[df['route_distance'] < 10000][df['duration'] < pd.Timedelta(1, 'h')]

df = tranform_data(bookings)

def view_data(df):
  plot = (p9.ggplot(df) + p9.geom_point(p9.aes(x='route_distance', y='duration', fill='tour_value')) +
          p9.ggtitle('Price, Distance, & Duration of Taxi-Fares (n=' + str(len(df.index)) + ')') +
    p9.xlab('Distance (m)') +
    p9.ylab(''))
  plot.save("model.png", dpi=1000)

view_data(df)