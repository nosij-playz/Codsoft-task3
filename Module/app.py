import pandas as pd
from chunkpredict import ChurnPredictor
predictor = ChurnPredictor("churn_model.pkl")

input_data = {
    "CreditScore": int(input("Enter Credit Score: ")),
    "Geography": input("Enter Geography (France/Germany/Spain): "),
    "Gender": input("Enter Gender (Male/Female): "),
    "Age": int(input("Enter Age: ")),
    "Tenure": int(input("Enter Tenure: ")),
    "Balance": float(input("Enter Balance: ")),
    "NumOfProducts": int(input("Enter Number of Products: ")),
    "HasCrCard": int(input("Has Credit Card (1 = Yes, 0 = No): ")),
    "IsActiveMember": int(input("Is Active Member (1 = Yes, 0 = No): ")),
    "EstimatedSalary": float(input("Enter Estimated Salary: "))
}

user_df = pd.DataFrame([input_data])

predictions, probs = predictor.predict(user_df)

user_df['PredictedChurn'] = predictions
user_df['ChurnProbability'] = probs
print("\nPrediction Result:")
print(user_df[['PredictedChurn', 'ChurnProbability']])
