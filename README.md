# Predicting Airfare Using Machine Learning

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
