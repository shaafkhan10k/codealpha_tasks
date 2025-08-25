"""
Task 1: Credit Scoring Model (CSV-based)
CodeAlpha Machine Learning Internship

Objective: Predict an individual's creditworthiness using past financial data
Approach: Classification algorithms (Logistic Regression, Decision Trees, Random Forest)
Key Features: Feature engineering, model evaluation with Precision, Recall, F1-Score, ROC-AUC

Author: CodeAlpha Intern
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CreditScoringModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_columns = []

    def load_csv_data(self, file_path):
        """Load dataset from CSV"""
        df = pd.read_csv(file_path)
        if 'creditworthy' not in df.columns:
            raise ValueError("Dataset must contain a 'creditworthy' target column.")
        return df

    def prepare_data(self, df):
        """Prepare features and target"""
        self.feature_columns = [col for col in df.columns if col != 'creditworthy']
        X = df[self.feature_columns]
        y = df['creditworthy']
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

    def initialize_models(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train models and calculate metrics"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            print(classification_report(y_test, y_pred))

            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC={self.results[name]['roc_auc']:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.show()

    def model_comparison(self):
        df = pd.DataFrame(self.results).T
        print("\nModel Comparison:\n", df)
        return df


def main():
    credit_model = CreditScoringModel()
    df = credit_model.load_csv_data("CodeAlpha_CreditScoringModel\credit_data.csv")
    X_train, X_test, y_train, y_test = credit_model.prepare_data(df)
    credit_model.initialize_models()
    credit_model.train_and_evaluate(X_train, X_test, y_train, y_test)
    results = credit_model.model_comparison()
    return credit_model, df, results


if __name__ == "__main__":
    model, dataset, results = main()
