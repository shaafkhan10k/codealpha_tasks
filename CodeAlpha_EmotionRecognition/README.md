# CodeAlpha\_EmotionRecognition

This project was completed as part of my **Machine Learning Internship with CodeAlpha**.
The goal is to build an **Emotion Recognition System** that classifies human emotions from speech signals.

---

##  Objective

Recognize emotions such as **happy, sad, angry, neutral, fear, disgust, surprise** from audio speech data.

---

## 🛠 Tech Stack

* Python 
* TensorFlow / Keras (Deep Learning Models: CNN, LSTM, GRU)
* NumPy, Pandas (data handling)
* Scikit-learn (evaluation metrics)
* Matplotlib / Seaborn (visualizations)

---

##  Dataset

* For demonstration, synthetic MFCC features were generated.
* Recommended real datasets:

  *  **[RAVDESS](https://zenodo.org/record/1188976)**
  *  **[TESS](https://tspace.library.utoronto.ca/handle/1807/24487)**
  *  **[EMO-DB](http://emodb.bilderbar.info/)**
* Features extracted: **MFCC (Mel-Frequency Cepstral Coefficients)**.

---

##  Approach

1. **Preprocessing**: Extract MFCC features from audio data
2. **Model Training**: Implemented 3 architectures

   * Convolutional Neural Network (CNN)
   * Long Short-Term Memory (LSTM)
   * Gated Recurrent Unit (GRU)
3. **Evaluation**: Metrics → Accuracy, Precision, Recall, F1-Score
4. **Testing**: Predictions on unseen data

---

##  Results (Sample Output with Synthetic Data)

| Model | Training Accuracy | Test Accuracy |
| ----- | ----------------- | ------------- |
| CNN   | 100%              | 100%          |
| LSTM  | 100%              | 99.6%         |
| GRU   | 100%              | 100%          |

 **All models achieved near-perfect accuracy on synthetic dataset.**
 For real datasets, results will be lower and require regularization & augmentation.

---

##  Repository Structure

```
CodeAlpha_EmotionRecognition/
│── task2_emotion_recognition.py   # Main script
│── README.md                      # Documentation

```

---

##  How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/shaafkhan10k/CodeAlpha_EmotionRecognition.git
   cd CodeAlpha_EmotionRecognition
   ```
2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:

   ```bash
   python task2_emotion_recognition.py
   ```

---
##  Demo Video
Watch the full walkthrough of Task 1 – Credit Scoring Model:

[Demo Video on Google Drive](https://drive.google.com/file/d/1Cyn6GvmO8H0z6O6pLaExZ_UqJ7xl9rxX/view?usp=sharing)


---

## Acknowledgments

* Internship provided by **[CodeAlpha](https://www.codealpha.tech/)**
* Emotion datasets: RAVDESS, TESS, EMO-DB
* Community and open-source contributors

---
