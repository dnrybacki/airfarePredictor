import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import plotly.express as px

# reg_model = joblib.load('/Users/dylan/code/Airfare Prediction Model/tuned_flight_price.pkl') #loads tuned random forrest model
reg_model = joblib.load('tuned_flight_price.pkl') #loads tuned random forrest model
df = pd.read_csv('FlightPricePrediction/Clean_Dataset.csv') #loads dataframe

# creates a list of the features from the model
feature_names = ['duration', 'days_left', 'source_city_Bangalore', 'source_city_Chennai',
                 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata',
                 'source_city_Mumbai', 'airline_AirAsia', 'airline_Air_India',
                 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
                 'departure_time_Afternoon', 'departure_time_Early_Morning',
                 'departure_time_Evening', 'departure_time_Late_Night',
                 'departure_time_Morning', 'departure_time_Night',
                 'destination_city_Bangalore', 'destination_city_Chennai',
                 'destination_city_Delhi', 'destination_city_Hyderabad',
                 'destination_city_Kolkata', 'destination_city_Mumbai', 'class_Business',
                 'class_Economy', 'DayOfWeek_Friday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday',
                 'DayOfWeek_Sunday', 'DayOfWeek_Thursday', 'DayOfWeek_Tuesday',
                 'DayOfWeek_Wednesday', 'nStops']

# Defines default values for the features in the UI
defaults = {
    'duration': int(df.duration.median()), #defaults to the median duration of flight
    'days_left': 1,
    'source_city': 'Bangalore',
    'airline': 'AirAsia',
    'departure_time': 'Morning',
    'destination_city': 'Chennai',
    'class': 'Economy',
    'nStops': 0,
    'DayOfWeek': 'Sunday'
}

# Creats list for dummy variables
source_cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
airlines = ['AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara']
departure_times = ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']
destination_cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
classes = ['Business', 'Economy']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# Creates a sidebar for input
st.sidebar.header("Input Features") #header text

source_city = st.sidebar.selectbox('Source City', source_cities, index=source_cities.index(defaults['source_city'])) #makes a selection box for the source city
destination_city = st.sidebar.selectbox('Destination City', destination_cities, index=destination_cities.index(defaults['destination_city'])) #same for destination city

airline = st.sidebar.selectbox('Airline', airlines, index=airlines.index(defaults['airline'])) #airline seleciton box
travel_class = st.sidebar.selectbox('Class', classes, index=classes.index(defaults['class'])) #class selection box


departure_time = st.sidebar.selectbox('Departure Time', departure_times, index=departure_times.index(defaults['departure_time'])) #departure time seleciton box

days_left = st.sidebar.number_input('Days Until Departure', 0, 51, defaults['days_left']) #days left input ranging from 0 to 51
duration = st.sidebar.slider('Duration (Hours)', 0, 50, defaults['duration']) #duration selection ranging from 0 to 50

nStops = st.sidebar.selectbox('Number of Stops', [0, 1, 2], index=defaults['nStops']) #number of stops selection box

DayOfWeek = st.sidebar.selectbox('Day of the Week', days_of_week, index=days_of_week.index(defaults['DayOfWeek']))

# Creates a feature vector for the input
input_features = {
    'duration': duration,
    'days_left': days_left,
    f'source_city_{source_city}': 1,
    f'airline_{airline}': 1,
    f'departure_time_{departure_time}': 1,
    f'destination_city_{destination_city}': 1,
    f'class_{travel_class}': 1,
    'nStops': nStops,
    f'DayOfWeek_{DayOfWeek}': 1
}

# Ensures all other dummy variables are set to 0
for feature in feature_names:
    if feature not in input_features:
        input_features[feature] = 0

# Creates an empty DataFrame with the correct columns
input_df = pd.DataFrame(columns=feature_names)
input_df = input_df.append(input_features, ignore_index=True)
input_df = input_df.fillna(0)


#Predict based on tuned Random Forest Model
prediction = reg_model.predict(input_df)

# Display the prediction
st.write("## Prediction")
st.write(f"Price in Rupees: {float(prediction):.2f}")

#Finding Optimal Booking Time
date_optimizer = pd.DataFrame({
    'days_left': range(1, 51),
    'predicted_price': [None] * 50
})

tempdf = input_df.copy() #creates a temporary data frame of the input

# iterates through each days until (from 1 to 51) to find the predicted best time to book and store it in the date_optimizer data frame
for i in range(1, 51):
    tempdf.days_left = i
    date_optimizer.loc[i - 1, 'predicted_price'] = reg_model.predict(tempdf)[0]

minPrice = date_optimizer.predicted_price.min() #stores predicted minimum price
minDate = date_optimizer.loc[date_optimizer['predicted_price'] == minPrice, 'days_left'].values[0] #stores days until for optimal booking time

#Displays optimal price/bookign time
st.write("### Optimal Booking Time:")
st.write(f"The optimal booking date given your parameters is {minDate} days before departure with a predicted price of {minPrice:.2f}")

st.write("#### Projected Price Over Time:")

#Displays interactive graph using plotly of predicted airfare based on given parameters.
date_optimizer['days_until'] = -1 * date_optimizer['days_left'] #inverts days_left for more intuitive graph


fig = px.line(date_optimizer, x='days_until', y='predicted_price', title='Predicted Trend in Price')
fig.update_layout(xaxis_title='Days Until Departure', yaxis_title='Predicted Price')
st.plotly_chart(fig)

