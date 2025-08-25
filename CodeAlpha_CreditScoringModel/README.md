# CodeAlpha\_CreditScoringModel

This project was completed as part of my **Machine Learning Internship with CodeAlpha**.
The goal is to build a **Credit Scoring Model** that predicts an individual's creditworthiness using financial data.

---

##  Objective

Predict whether a person is **creditworthy (1)** or **not creditworthy (0)** based on their past financial records.

---

## 🛠 Tech Stack

* Python 
* Pandas, NumPy (data preprocessing)
* Scikit-learn (Logistic Regression, Decision Tree, Random Forest)
* Matplotlib, Seaborn (visualization)

---

## Dataset

* Dataset: **credit\_data.csv** (synthetic dataset of 2000 rows created for this task)
* Features include:

  * Income, Age, Employment Years, Debt-to-Income Ratio
  * Credit History, Previous Defaults, Loan Amount, Savings, Expenses
  * Engineered Features (Loan-to-Income Ratio, Disposable Income, etc.)
* Target variable:

  * `creditworthy` → 1 = Good credit, 0 = Risky

---

##  Approach

1. Load and preprocess dataset
2. Feature engineering and scaling
3. Train models: Logistic Regression, Decision Tree, Random Forest
4. Evaluate using metrics:

   * Accuracy
   * Precision
   * Recall
   * F1-Score
   * ROC-AUC

---

##  Results (Sample)

| Model               | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  |
| ------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.87     | 0.84      | 0.80     | 0.82     | 0.88     |
| Decision Tree       | 0.85     | 0.81      | 0.83     | 0.82     | 0.85     |
| Random Forest       | **0.91** | **0.89**  | **0.87** | **0.88** | **0.93** |

 **Random Forest performed best overall.**

---

##  Repository Structure

```
CodeAlpha_CreditScoringModel/
│── credit_data.csv          # Dataset
│── task1_credit_scoring.py  # Main code
│── README.md                # Project documentation
```

---

##  How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/shaafkhan10k/CodeAlpha_CreditScoringModel.git
   cd CodeAlpha_CreditScoringModel
   ```
2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:

   ```bash
   python task1_credit_scoring.py
   ```

---
##  Demo Video
View the complete Task 2 walkthrough – Emotion Recognition from Speech:

[Demo Video on Google Drive](https://drive.google.com/file/d/1Cyn6GvmO8H0z6O6pLaExZ_UqJ7xl9rxX/view?usp=sharing)

---

##  Acknowledgments

* Internship provided by **[CodeAlpha](https://www.codealpha.tech/)**
* Mentors & community support

---
