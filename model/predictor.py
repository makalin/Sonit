"""
Sound predictor module for Sonit
Uses trained models to predict vocal gestures from audio features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for sound classification"""
    
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SoundPredictor:
    """Predictor for vocal gesture classification"""
    
    def __init__(self, model_type='neural_network'):
        """
        Initialize sound predictor
        
        Args:
            model_type (str): Type of model ('neural_network', 'random_forest', 'svm', 'mlp')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Model parameters
        self.input_size = None
        self.num_classes = None
        self.hidden_size = 128
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_size = 100
    
    def create_model(self, input_size, num_classes):
        """
        Create a new model
        
        Args:
            input_size (int): Number of input features
            num_classes (int): Number of output classes
        """
        self.input_size = input_size
        self.num_classes = num_classes
        
        if self.model_type == 'neural_network':
            self.model = SimpleNeuralNetwork(input_size, self.hidden_size, num_classes)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, features, labels, validation_split=0.2):
        """
        Train the model
        
        Args:
            features (list): List of feature vectors
            labels (list): List of corresponding labels
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            dict: Training metrics
        """
        if not features or not labels:
            raise ValueError("Features and labels cannot be empty")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create model if not exists
        if self.model is None:
            self.create_model(X.shape[1], len(self.label_encoder.classes_))
        
        # Train model
        if self.model_type == 'neural_network':
            return self._train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            return self._train_sklearn_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network model"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 100
        batch_size = 32
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = accuracy_score(y_val, val_predictions.numpy())
            
            train_losses.append(total_loss / (len(X_train) // batch_size))
            val_accuracies.append(val_accuracy)
            
            # Early stopping
            if epoch > 10 and val_accuracies[-1] < max(val_accuracies[-10:]):
                break
        
        self.is_fitted = True
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_val_accuracy': val_accuracies[-1]
        }
    
    def _train_sklearn_model(self, X_train, y_train, X_val, y_val):
        """Train sklearn model"""
        # Train model
        self.model.fit(X_train, y_train)
        
        # Validation
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        self.is_fitted = True
        
        return {
            'val_accuracy': val_accuracy,
            'classification_report': classification_report(y_val, y_val_pred)
        }
    
    def predict(self, features, return_confidence=True):
        """
        Predict class for given features
        
        Args:
            features (numpy.ndarray): Feature vector
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            tuple: (prediction, confidence) or just prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Check cache
        feature_hash = hash(features.tobytes())
        if feature_hash in self.prediction_cache:
            return self.prediction_cache[feature_hash]
        
        # Preprocess features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        if self.model_type == 'neural_network':
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prediction_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction_idx].item()
        else:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)
                prediction_idx = np.argmax(probabilities)
                confidence = probabilities[0][prediction_idx]
            else:
                prediction_idx = self.model.predict(features_scaled)[0]
                confidence = 1.0  # Default confidence for models without probability
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        # Cache result
        result = (prediction, confidence) if return_confidence else prediction
        self.prediction_cache[feature_hash] = result
        
        # Manage cache size
        if len(self.prediction_cache) > self.cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.prediction_cache.keys())[:self.cache_size // 2]
            for key in keys_to_remove:
                del self.prediction_cache[key]
        
        return result
    
    def predict_batch(self, features_list):
        """
        Predict classes for a batch of features
        
        Args:
            features_list (list): List of feature vectors
            
        Returns:
            list: List of (prediction, confidence) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Preprocess features
        features_array = np.array(features_list)
        features_scaled = self.scaler.transform(features_array)
        
        # Make predictions
        if self.model_type == 'neural_network':
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prediction_indices = torch.argmax(outputs, dim=1).numpy()
                confidences = torch.max(probabilities, dim=1)[0].numpy()
        else:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)
                prediction_indices = np.argmax(probabilities, axis=1)
                confidences = np.max(probabilities, axis=1)
            else:
                prediction_indices = self.model.predict(features_scaled)
                confidences = np.ones(len(features_list))
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(prediction_indices)
        
        return list(zip(predictions, confidences))
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'neural_network':
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.input_size,
                'num_classes': self.num_classes,
                'hidden_size': self.hidden_size,
                'model_type': self.model_type
            }, filepath)
            
            # Save preprocessing components
            preprocess_path = filepath.replace('.pth', '_preprocess.pkl')
            with open(preprocess_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder
                }, f)
        else:
            # Save sklearn model
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type
            }, filepath)
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if filepath.endswith('.pth'):
            # Load PyTorch model
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.input_size = checkpoint['input_size']
            self.num_classes = checkpoint['num_classes']
            self.hidden_size = checkpoint['hidden_size']
            self.model_type = checkpoint['model_type']
            
            self.create_model(self.input_size, self.num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load preprocessing components
            preprocess_path = filepath.replace('.pth', '_preprocess.pkl')
            if os.path.exists(preprocess_path):
                with open(preprocess_path, 'rb') as f:
                    preprocess_data = pickle.load(f)
                    self.scaler = preprocess_data['scaler']
                    self.label_encoder = preprocess_data['label_encoder']
        else:
            # Load sklearn model
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
        
        self.is_fitted = True
    
    def get_model_info(self):
        """Get information about the current model"""
        if not self.is_fitted:
            return {"status": "Not trained"}
        
        info = {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "classes": list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else [],
            "is_fitted": self.is_fitted
        }
        
        if self.model_type == 'neural_network':
            info["hidden_size"] = self.hidden_size
            info["total_parameters"] = sum(p.numel() for p in self.model.parameters())
        
        return info
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear() 