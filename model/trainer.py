"""
Model trainer module for Sonit
Handles training and evaluation of sound classification models
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from model.predictor import SoundPredictor, SimpleNeuralNetwork


class ModelTrainer:
    """Trainer for sound classification models"""
    
    def __init__(self, model_type='neural_network'):
        """
        Initialize model trainer
        
        Args:
            model_type (str): Type of model to train
        """
        self.model_type = model_type
        self.predictor = SoundPredictor(model_type)
        self.training_history = []
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.validation_split = 0.2
        self.early_stopping_patience = 10
        
        # Results storage
        self.best_model_path = None
        self.training_metrics = {}
    
    def train(self, features, labels, model_save_path=None, **kwargs):
        """
        Train a sound classification model
        
        Args:
            features (list): List of feature vectors
            labels (list): List of corresponding labels
            model_save_path (str): Path to save the trained model
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results and metrics
        """
        if not features or not labels:
            raise ValueError("Features and labels cannot be empty")
        
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")
        
        # Update parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        print(f"Starting training with {len(features)} samples...")
        print(f"Model type: {self.model_type}")
        print(f"Number of classes: {len(set(labels))}")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Train model
        if self.model_type == 'neural_network':
            results = self._train_neural_network(X_train, y_train, X_val, y_val)
        else:
            results = self._train_sklearn_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_predictions, val_confidences = zip(*[
            self.predictor.predict(feature) for feature in X_val
        ])
        
        val_accuracy = accuracy_score(y_val, val_predictions)
        results['validation_accuracy'] = val_accuracy
        results['validation_report'] = classification_report(y_val, val_predictions)
        
        # Save model if path provided
        if model_save_path:
            self.predictor.save_model(model_save_path)
            self.best_model_path = model_save_path
            results['model_saved'] = model_save_path
        
        # Store training metrics
        self.training_metrics = results
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'num_samples': len(features),
            'num_classes': len(set(labels)),
            'validation_accuracy': val_accuracy,
            'results': results
        })
        
        print(f"Training completed!")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        return results
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network model"""
        # Create model
        input_size = X_train.shape[1]
        num_classes = len(set(y_train))
        self.predictor.create_model(input_size, num_classes)
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.predictor.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.predictor.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.predictor.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.predictor.model.eval()
            with torch.no_grad():
                val_outputs = self.predictor.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = accuracy_score(y_val, val_predictions.numpy())
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_accuracy:.4f}")
        
        self.predictor.is_fitted = True
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'num_epochs_trained': len(train_losses)
        }
    
    def _train_sklearn_model(self, X_train, y_train, X_val, y_val):
        """Train sklearn model"""
        # Create and train model
        self.predictor.train(X_train.tolist(), y_train.tolist(), validation_split=0.0)
        
        # Evaluate on validation set
        val_predictions = []
        for feature in X_val:
            prediction, _ = self.predictor.predict(feature)
            val_predictions.append(prediction)
        
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        return {
            'validation_accuracy': val_accuracy,
            'validation_report': classification_report(y_val, val_predictions)
        }
    
    def cross_validate(self, features, labels, cv_folds=5):
        """
        Perform cross-validation
        
        Args:
            features (list): List of feature vectors
            labels (list): List of corresponding labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        if self.model_type == 'neural_network':
            # For neural networks, we'll do manual cross-validation
            return self._cross_validate_neural_network(features, labels, cv_folds)
        else:
            # For sklearn models, use built-in cross-validation
            return self._cross_validate_sklearn(features, labels, cv_folds)
    
    def _cross_validate_neural_network(self, features, labels, cv_folds):
        """Manual cross-validation for neural networks"""
        X = np.array(features)
        y = np.array(labels)
        
        fold_scores = []
        fold_reports = []
        
        # Manual k-fold cross-validation
        fold_size = len(X) // cv_folds
        
        for fold in range(cv_folds):
            # Create fold indices
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < cv_folds - 1 else len(X)
            
            # Split data
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            X_train = np.concatenate([X[:val_start], X[val_end:]])
            y_train = np.concatenate([y[:val_start], y[val_end:]])
            
            # Train model
            self.train(X_train.tolist(), y_train.tolist())
            
            # Evaluate
            val_predictions = []
            for feature in X_val:
                prediction, _ = self.predictor.predict(feature)
                val_predictions.append(prediction)
            
            fold_accuracy = accuracy_score(y_val, val_predictions)
            fold_scores.append(fold_accuracy)
            fold_reports.append(classification_report(y_val, val_predictions))
        
        return {
            'fold_scores': fold_scores,
            'mean_accuracy': np.mean(fold_scores),
            'std_accuracy': np.std(fold_scores),
            'fold_reports': fold_reports
        }
    
    def _cross_validate_sklearn(self, features, labels, cv_folds):
        """Cross-validation for sklearn models"""
        X = np.array(features)
        y = np.array(labels)
        
        # Create temporary predictor for cross-validation
        temp_predictor = SoundPredictor(self.model_type)
        temp_predictor.create_model(X.shape[1], len(set(y)))
        
        # Perform cross-validation
        scores = cross_val_score(temp_predictor.model, X, y, cv=cv_folds, scoring='accuracy')
        
        return {
            'fold_scores': scores.tolist(),
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
    
    def evaluate_model(self, features, labels, save_confusion_matrix=True):
        """
        Evaluate trained model on test data
        
        Args:
            features (list): Test feature vectors
            labels (list): Test labels
            save_confusion_matrix (bool): Whether to save confusion matrix plot
            
        Returns:
            dict: Evaluation results
        """
        if not self.predictor.is_fitted:
            raise RuntimeError("Model must be trained before evaluation")
        
        X_test = np.array(features)
        y_test = np.array(labels)
        
        # Make predictions
        predictions = []
        confidences = []
        
        for feature in X_test:
            prediction, confidence = self.predictor.predict(feature)
            predictions.append(prediction)
            confidences.append(confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # Save confusion matrix if requested
        if save_confusion_matrix:
            self._save_confusion_matrix(conf_matrix, y_test, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': predictions,
            'confidences': confidences,
            'num_samples': len(features)
        }
        
        return results
    
    def _save_confusion_matrix(self, conf_matrix, y_true, y_pred):
        """Save confusion matrix as a plot"""
        try:
            # Get unique labels
            labels = sorted(list(set(y_true) | set(y_pred)))
            
            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save plot
            os.makedirs('data/plots', exist_ok=True)
            plot_path = f'data/plots/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix saved to: {plot_path}")
            
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
    
    def save_training_history(self, filepath):
        """Save training history to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self, filepath):
        """Load training history from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.training_history = json.load(f)
    
    def get_training_summary(self):
        """Get summary of training history"""
        if not self.training_history:
            return {"message": "No training history available"}
        
        summary = {
            "total_training_sessions": len(self.training_history),
            "latest_training": self.training_history[-1],
            "best_accuracy": max([h['validation_accuracy'] for h in self.training_history]),
            "average_accuracy": np.mean([h['validation_accuracy'] for h in self.training_history])
        }
        
        return summary 