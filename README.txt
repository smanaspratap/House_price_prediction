House Price Prediction using Machine Learning

Project Overview
This project aims to predict house prices using various machine learning models such as:
- Linear Regression
- Decision Tree
- Random Forest

The dataset used is the **California Housing Dataset** from Scikit-learn, containing features such as:
- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude & Longitude

Steps Followed

 1. Dataset Selection
- Used the California Housing Dataset from Scikit-learn.
- Loaded the dataset using Pandas and displayed relevant information.

 2. Data Preprocessing
- Checked for missing values (none found).
- Scaled features using `StandardScaler` for better model performance.
- Split the data into training (80%) and testing (20%) sets.

 3. Model Selection & Training
- Implemented Linear Regression, Decision Tree, and Random Forest.
- Trained all models on the training dataset.

 4. Model Evaluation
- Evaluated each model's performance using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

 5. Hyperparameter Tuning
- Performed tuning on Decision Tree and Random Forest models to improve performance.

 6. Visualization
- Visualized predicted vs actual house prices for all models.
- Created a residual plot to analyze model errors.

Results
- Linear Regression: MAE: 0.5332 | RMSE: 0.7456
- Decision Tree: MAE: 0.5008 | RMSE: 0.7052
- Random Forest: MAE: 0.4608 | RMSE: 0.6482

How to Run the Code
1. Clone the repository.
2. Install dependencies using:

pip install -r requirements.txt

3. Run the following command:

python main.py


Libraries Required
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

Insights
- Random Forest performed the best in terms of both MAE and RMSE, making it the most reliable model for this dataset.
- Visualizing predicted vs actual prices helped identify potential overfitting issues in some models.

