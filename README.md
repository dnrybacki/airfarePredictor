# Predicting Airfare Using Machine Learning

A non-interactive version of the project can be viewed here: https://htmlpreview.github.io/?https://github.com/dnrybacki/airfarePredictor/blob/82c479754c9e645ab600ce9d074589139e486ace/AirfarePrediction%20NonInteractive.html. (It will take a few moments to load)

To view the interactive graphs, download 'AirfarePrediction.ipynb'

## Project Overview

This project is an attempt to predict the airfare of flights, first by exploring the data set, preprocessing the data set for predictive models, finding an ideal model, and creating a user interface for the model's predictions in hopes to extract insights into the aviation industry and how airlines price their flights. 

#### 1. Data Exploration

- Dataset will be analyzed for features present
- Explores statistics to gain understanding of the distribution of some features

#### 2. Data Visualization

- Dataset will be further explored with the graphical help of matplotlib, seaborne, and plotly
- Derive insights from the dataset
- Explore the relationship between features

#### 3. Pre-Processing and Feature Engineering
- Identifying which features are to be predictor features of price
- Engineering a day of the week feature
- One hot encoding categorical data
- Display feature coorelation table
- Splitting data into training and testing

#### 4. Modeling
- Testing performace of linear regression on data
- Analyze other models such as Lasso, Ridge, Decision Tree, Random Forrest, and XGBoost
- Determine best model using $R^2$ and $RMSE$ and hyperparameter tuning to optimize it
- Save model to be used in user interface

#### 5. Conclusion


## Using the User Interface
- Run through the code in the notebook (most importantly the data import, preprocessing, and the final tuned model)
- Model should be in a pickle file in the same folder
- Run the commond 'streamlit run app.py' in terminal with your working directory set as the folder of this project
- UI will allow you to change the values of the features which will then show you the predicted price given those inputs and also the ideal booking time with a visual represntation for how the model predicts the price will change over time.

<img width="1624" alt="Screenshot 2024-08-13 at 11 43 21 PM" src="https://github.com/user-attachments/assets/39ee74ae-87ee-45f0-8cef-4ebde4d12a38">
