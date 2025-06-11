import pandas as pd
import joblib

class ChurnPredictor:
    def __init__(self, model_path="churn_model.pkl"):
        self.model = joblib.load(model_path)

    def preprocess(self, df):
        # Basic preprocessing
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

        # Ensure all expected one-hot encoded columns are present
        for col in ['Geography_Germany', 'Geography_Spain']:
            if col not in df.columns:
                df[col] = 0

        # Feature engineering (must match training time)
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
        df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
        df['TenureByAge'] = df['Tenure'] / df['Age']
        df['CreditScorePerAge'] = df['CreditScore'] / df['Age']
        df['IsHighValueCustomer'] = (
            (df['Balance'] > df['Balance'].median()) &
            (df['EstimatedSalary'] > df['EstimatedSalary'].median())
        ).astype(int)
        df['Products_CreditCard'] = df['NumOfProducts'] * df['HasCrCard']
        df['ActiveHighBalance'] = (
            (df['IsActiveMember'] == 1) &
            (df['Balance'] > df['Balance'].median())
        ).astype(int)

        df.fillna(0, inplace=True)

        # Drop columns that the model wasn’t trained with
        drop_cols = [col for col in ["RowNumber", "CustomerId", "Surname", "Exited"] if col in df.columns]
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return df

    def predict(self, df):
        df_processed = self.preprocess(df.copy())
        predictions = self.model.predict(df_processed)
        prediction_probs = self.model.predict_proba(df_processed)[:, 1]
        return predictions, prediction_probs
