# Housing Price Prediction using Random Forests model

## Steps taken:
  * Uploaded "Housing" dataset from Kaggle
  * Carried out an Exploratory Data Analysis on the dataset using **_Pandas_** and **_numpy_**
    - calculated summary statistics for the features to learn more about the data
    - checked for outliers and missing values
    - transformed the data, specifically the categorical features so that they can be used in the predictive model
  * Utilized **_seaborn_** to visualize the features using _scatter plot_ and _bar plot_
  * Generated a _correlation matrix_ to identify the features that are of significance to the study variable (price)
  * Utilized **_scikitlearn_** library to:
    - Divide the dataset into test and validation subsets
    - Fit a **_Random Forest Regressor_** model to the test subset with ```random_state=1```
    - Predict the price for the validation subset using the above fitted model
  * Calculated the accuracy score, mean absolute error, R-squared value and out-of-bag score to measure the precision of the model
## Outcome:
  * Identified categorical variables that needed dummy coding (0's and 1's)
  * Identified the explanatory variables that are highly correlated with study variable (price) and used them as the features for model generation
    ```
     features=["area","bedrooms","bathrooms","stories","mainroad","guestroom","airconditioning","parking","prefarea","furnishingstatus"]
    ```
  *  Fit the model using Random Forests with an accuracy score of 0.61 and R^2 value of 0.30
