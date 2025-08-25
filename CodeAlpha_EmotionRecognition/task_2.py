"""
Task 2: Emotion Recognition from Speech
CodeAlpha Machine Learning Internship

Objective: Recognize human emotions (happy, angry, sad, neutral, etc.) from speech audio
Approach: Deep learning and speech signal processing with MFCC features
Key Features: Feature extraction (MFCCs), CNN/RNN/LSTM models, emotion classification

Author: CodeAlpha Intern  
Date: August 2025

Note: This implementation includes synthetic data generation for demonstration.
For real implementation, use datasets like RAVDESS, TESS, or EMO-DB.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, GRU
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class EmotionRecognitionModel:
    """
    Emotion Recognition from Speech using Deep Learning
    Implements CNN, RNN, and LSTM models for emotion classification
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.history = {}
        self.emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise']
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def generate_synthetic_mfcc_features(self, n_samples=2000, n_mfcc=13, n_frames=100):
        """
        Generate synthetic MFCC features for demonstration
        In real implementation, extract these from audio files using librosa
        """
        print("Generating synthetic MFCC features...")

        # Create realistic MFCC-like features for different emotions
        features = []
        labels = []

        for emotion in self.emotion_labels:
            samples_per_emotion = n_samples // len(self.emotion_labels)

            for i in range(samples_per_emotion):
                # Generate emotion-specific MFCC patterns
                if emotion == 'angry':
                    # Higher energy, more variation
                    mfcc = np.random.normal(0, 2, (n_frames, n_mfcc))
                    mfcc[:, 0] += np.random.normal(5, 1)  # Higher energy in first coefficient
                elif emotion == 'happy':
                    # Higher pitch variation, positive energy
                    mfcc = np.random.normal(1, 1.5, (n_frames, n_mfcc))
                    mfcc[:, 1:3] += np.random.normal(2, 0.5, (n_frames, 2))
                elif emotion == 'sad':
                    # Lower energy, less variation
                    mfcc = np.random.normal(-1, 1, (n_frames, n_mfcc))
                    mfcc[:, 0] -= np.random.normal(3, 0.5)
                elif emotion == 'neutral':
                    # Balanced, moderate variation
                    mfcc = np.random.normal(0, 1, (n_frames, n_mfcc))
                elif emotion == 'fear':
                    # High variation, trembling effect
                    mfcc = np.random.normal(0, 2.5, (n_frames, n_mfcc))
                    # Add trembling pattern
                    for j in range(n_frames):
                        if j % 10 < 5:
                            mfcc[j] += np.random.normal(0, 0.5, n_mfcc)
                elif emotion == 'disgust':
                    # Negative bias with specific patterns
                    mfcc = np.random.normal(-0.5, 1.2, (n_frames, n_mfcc))
                    mfcc[:, 2:4] -= np.random.normal(1, 0.3, (n_frames, 2))
                else:  # surprise
                    # Sharp changes, high variation in specific coefficients
                    mfcc = np.random.normal(0, 1.8, (n_frames, n_mfcc))
                    # Add sudden spikes
                    spike_frames = np.random.choice(n_frames, size=10, replace=False)
                    mfcc[spike_frames] += np.random.normal(3, 1, (10, n_mfcc))

                features.append(mfcc)
                labels.append(emotion)

        return np.array(features), np.array(labels)

    def preprocess_data(self, features, labels):
        """Preprocess MFCC features and labels"""
        print("Preprocessing data...")

        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_categorical, test_size=0.2, 
            random_state=self.random_state, stratify=labels_encoded
        )

        # Normalize features
        # Reshape for scaling
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)

        # Reshape back
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        print(f"Training data shape: {X_train_scaled.shape}")
        print(f"Test data shape: {X_test_scaled.shape}")
        print(f"Number of emotion classes: {len(self.emotion_labels)}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_cnn_model(self, input_shape, num_classes):
        """Build CNN model for emotion recognition"""
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm_model(self, input_shape, num_classes):
        """Build LSTM model for emotion recognition"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_gru_model(self, input_shape, num_classes):
        """Build GRU model for emotion recognition"""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
        """Train all models"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(self.emotion_labels)

        print(f"\nTraining models with input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")

        # Build models
        self.models['CNN'] = self.build_cnn_model(input_shape, num_classes)
        self.models['LSTM'] = self.build_lstm_model(input_shape, num_classes)
        self.models['GRU'] = self.build_gru_model(input_shape, num_classes)

        # Train each model
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name} Model")
            print('='*50)

            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=0  # Set to 1 to see training progress
            )

            self.history[name] = history

            # Evaluate model
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")

            # Predictions for classification report
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)

            print(f"\nClassification Report for {name}:")
            print(classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.emotion_labels, zero_division=0
            ))

    def get_model_summary(self):
        """Print summary of all models"""
        print(f"\n{'='*60}")
        print("MODEL ARCHITECTURE SUMMARY")
        print('='*60)

        for name, model in self.models.items():
            print(f"\n{name} Model:")
            print("-" * 30)
            model.summary()

    def predict_emotion(self, mfcc_features, model_name='CNN'):
        """Predict emotion from MFCC features"""
        model = self.models[model_name]

        # Preprocess input
        features_scaled = self.scaler.transform(mfcc_features.reshape(1, -1))
        features_scaled = features_scaled.reshape(1, mfcc_features.shape[0], mfcc_features.shape[1])

        # Predict
        prediction = model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        emotion = self.label_encoder.inverse_transform([predicted_class])[0]

        return emotion, confidence, prediction[0]


def extract_mfcc_features_real(audio_file_path, n_mfcc=13):
    """
    Real MFCC extraction function (requires librosa)
    Uncomment and use this for real audio files

    import librosa

    # Load audio file
    audio, sr = librosa.load(audio_file_path, sr=22050)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Transpose to get (time_frames, n_mfcc) shape
    mfccs = mfccs.T

    return mfccs
    """
    pass


def main():
    """Main function to run emotion recognition model"""
    print("="*60)
    print("EMOTION RECOGNITION FROM SPEECH - CODEALPHA INTERNSHIP")
    print("="*60)

    # Initialize model
    emotion_model = EmotionRecognitionModel()

    # Generate synthetic MFCC features
    print("\n1. Generating synthetic MFCC features...")
    features, labels = emotion_model.generate_synthetic_mfcc_features(n_samples=1400)
    print(f"Generated {len(features)} samples")
    print(f"Feature shape per sample: {features[0].shape}")
    print(f"Emotions: {emotion_model.emotion_labels}")

    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test = emotion_model.preprocess_data(features, labels)

    # Train models
    print("\n3. Training deep learning models...")
    emotion_model.train_models(X_train, X_test, y_train, y_test, epochs=30)

    # Model summaries
    print("\n4. Model Architecture Summary:")
    emotion_model.get_model_summary()

    # Test prediction
    print("\n5. Testing Emotion Prediction:")
    print("="*50)
    test_sample = X_test[0]
    true_emotion_idx = np.argmax(y_test[0])
    true_emotion = emotion_model.emotion_labels[true_emotion_idx]

    for model_name in ['CNN', 'LSTM', 'GRU']:
        predicted_emotion, confidence, probabilities = emotion_model.predict_emotion(
            test_sample, model_name
        )
        print(f"{model_name} Model:")
        print(f"  True Emotion: {true_emotion}")
        print(f"  Predicted Emotion: {predicted_emotion}")
        print(f"  Confidence: {confidence:.4f}")
        print()

    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR REAL IMPLEMENTATION")
    print("="*60)
    print("1. Use real audio datasets: RAVDESS, TESS, EMO-DB")
    print("2. Install librosa for real MFCC extraction: pip install librosa")
    print("3. Implement data augmentation for better performance")
    print("4. Consider combining multiple features (MFCC + spectral features)")
    print("5. Use early stopping and model checkpointing for training")
    print("6. Implement cross-validation for robust evaluation")

    return emotion_model


if __name__ == "__main__":
    # Run the complete emotion recognition analysis
    model = main()

    print("\n" + "="*60)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNote: This implementation uses synthetic data for demonstration.")
    print("For production use, replace with real audio processing using librosa.")
