"""
Complete Generator Modules
Deep Learning, NLP, RL, DSA, and Web Generators
"""

from datetime import datetime
import re

# ========================================
# DEEP LEARNING GENERATOR
# ========================================
class DLGenerator:
    """Generate Deep Learning code with TensorFlow/PyTorch"""
    
    def generate(self, project_name, description, options):
        """Generate DL project code"""
        architecture = options.get('architecture', 'CNN')
        framework = options.get('framework', 'tensorflow')
        
        if framework == 'tensorflow':
            main_code = self._generate_tensorflow_code(project_name, description, architecture)
        else:
            main_code = self._generate_pytorch_code(project_name, description, architecture)
        
        return {
            'main_code': main_code,
            'utils_code': self._generate_dl_utils(),
            'requirements': self._generate_dl_requirements(framework),
            'readme': self._generate_dl_readme(project_name, description, architecture)
        }
    
    def _generate_tensorflow_code(self, project_name, description, architecture):
        class_name = self._to_class_name(project_name)
        
        return f'''"""
{project_name} - Deep Learning Model
{description}

Framework: TensorFlow/Keras
Architecture: {architecture}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


class {class_name}:
    """Deep Learning model for {project_name}"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.data_augmentation = self._create_data_augmentation()
    
    def build_model(self):
        """Build {architecture} architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Convolutional blocks
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def _create_data_augmentation(self):
        """Create data augmentation pipeline"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        print(f"Training {{self.__class__.__name__}}...")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            self.data_augmentation.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self._plot_training_history()
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\\nEvaluating model...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        print(f"\\nTest Loss: {{results[0]:.4f}}")
        print(f"Test Accuracy: {{results[1]:.4f}}")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes, predictions
    
    def _plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.close()
        print("Training history saved to training_history.png")
    
    def save_model(self, filepath='model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {{filepath}}")
    
    def load_model(self, filepath='model.h5'):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {{filepath}}")


def main():
    """Main execution"""
    print("="*70)
    print(f"  {project_name.upper()}")
    print("="*70)
    
    # Initialize model
    model = {class_name}(input_shape=(224, 224, 3), num_classes=10)
    model.build_model()
    
    print(model.model.summary())
    
    # Note: Add your data loading and training code here
    # model.train(X_train, y_train, X_val, y_val)
    # model.evaluate(X_test, y_test)
    # model.save_model()


if __name__ == "__main__":
    main()
'''
    
    def _generate_dl_utils(self):
        return '''"""
Deep Learning Utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()


def visualize_predictions(images, true_labels, pred_labels, class_names, n=10):
    """Visualize predictions"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx in range(min(n, len(images))):
        axes[idx].imshow(images[idx])
        true_name = class_names[true_labels[idx]] if class_names else true_labels[idx]
        pred_name = class_names[pred_labels[idx]] if class_names else pred_labels[idx]
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        axes[idx].set_title(f'True: {true_name}\\nPred: {pred_name}', color=color)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300)
    plt.close()
'''
    
    def _generate_dl_requirements(self, framework):
        reqs = [
            "numpy==1.24.3",
            "matplotlib==3.7.2",
            "seaborn==0.12.2",
            "scikit-learn==1.3.0"
        ]
        
        if framework == 'tensorflow':
            reqs.append("tensorflow==2.13.0")
        else:
            reqs.append("torch==2.0.1")
            reqs.append("torchvision==0.15.2")
        
        return "\n".join(reqs)
    
    def _generate_dl_readme(self, project_name, description, architecture):
        return f"""# {project_name}

{description}

## Architecture
{architecture} - Convolutional Neural Network

## Features
- Data augmentation
- Batch normalization
- Dropout regularization
- Learning rate scheduling
- Early stopping
- Model checkpointing

## Usage
```python
from main import {self._to_class_name(project_name)}

model = {self._to_class_name(project_name)}()
model.build_model()
model.train(X_train, y_train, X_val, y_val)
```
"""
    
    @staticmethod
    def _to_class_name(name):
        return ''.join(word.capitalize() for word in re.sub(r'[^a-zA-Z0-9\s]', '', name).split())


# ========================================
# NLP GENERATOR
# ========================================
class NLPGenerator:
    """Generate NLP code with Transformers"""
    
    def generate(self, project_name, description, options):
        task = options.get('task', 'Sentiment Analysis')
        
        return {
            'main_code': self._generate_nlp_code(project_name, description, task),
            'utils_code': self._generate_nlp_utils(),
            'requirements': "transformers==4.30.0\\ntorch==2.0.1\\npandas==2.0.3\\nnumpy==1.24.3",
            'readme': f"# {project_name}\\n\\n{description}\\n\\nNLP Task: {task}"
        }
    
    def _generate_nlp_code(self, project_name, description, task):
        return f'''"""
{project_name} - NLP Model
{description}
Task: {task}
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np


class NLPModel:
    """NLP model using transformers"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def preprocess_texts(self, texts, labels=None):
        """Tokenize and prepare texts"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        if labels is not None:
            dataset = Dataset.from_dict({{
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            }})
        else:
            dataset = Dataset.from_dict({{
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            }})
        
        return dataset
    
    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=3):
        """Train the model"""
        train_dataset = self.preprocess_texts(train_texts, train_labels)
        val_dataset = self.preprocess_texts(val_texts, val_labels)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        return trainer
    
    def predict(self, texts):
        """Make predictions"""
        self.model.eval()
        dataset = self.preprocess_texts(texts)
        
        predictions = []
        with torch.no_grad():
            for item in dataset:
                inputs = {{k: v.unsqueeze(0).to(self.device) for k, v in item.items()}}
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1)
                predictions.append(pred.item())
        
        return predictions


def main():
    print("NLP Model Initialized")
    model = NLPModel()
    # Add your training code here


if __name__ == "__main__":
    main()
'''
    
    def _generate_nlp_utils(self):
        return "# NLP utility functions\\n"


# ========================================
# RL GENERATOR
# ========================================
class RLGenerator:
    """Generate Reinforcement Learning code"""
    
    def generate(self, project_name, description, options):
        algorithm = options.get('algorithm', 'Q-Learning')
        
        return {
            'main_code': self._generate_rl_code(project_name, algorithm),
            'utils_code': "# RL utilities",
            'requirements': "gym==0.26.0\\nstable-baselines3==2.0.0\\nnumpy==1.24.3",
            'readme': f"# {project_name}\\n\\n{description}\\n\\nAlgorithm: {algorithm}"
        }
    
    def _generate_rl_code(self, project_name, algorithm):
        return f'''"""
{project_name} - Reinforcement Learning
Algorithm: {algorithm}
"""

import gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env


class RLAgent:
    """Reinforcement Learning Agent"""
    
    def __init__(self, env_name='CartPole-v1'):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.model = None
    
    def create_model(self, algorithm='PPO'):
        """Create RL model"""
        if algorithm == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif algorithm == 'DQN':
            self.model = DQN('MlpPolicy', self.env, verbose=1)
        return self.model
    
    def train(self, total_timesteps=10000):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps)
    
    def evaluate(self, n_episodes=10):
        """Evaluate agent"""
        total_rewards = []
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards)


def main():
    agent = RLAgent()
    agent.create_model()
    agent.train(10000)
    mean_reward = agent.evaluate()
    print(f"Mean reward: {{mean_reward}}")


if __name__ == "__main__":
    main()
'''


# ========================================
# DSA SOLVER
# ========================================
class DSASolver:
    """Generate DSA solutions"""
    
    def generate(self, project_name, description, options):
        problem_type = options.get('problem_type', 'Arrays')
        
        return {
            'main_code': self._generate_dsa_code(project_name, description, problem_type),
            'utils_code': "# DSA utility functions",
            'requirements': "# No special requirements needed",
            'readme': f"# {project_name}\\n\\n{description}\\n\\nProblem Type: {problem_type}"
        }
    
    def _generate_dsa_code(self, project_name, description, problem_type):
        return f'''"""
{project_name} - DSA Solution
{description}
Problem Type: {problem_type}
"""

from typing import List, Optional


class Solution:
    """DSA Problem Solution"""
    
    def solve(self, input_data):
        """
        Main solution method
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        # Implement your solution here
        pass
    
    def validate_input(self, input_data):
        """Validate input data"""
        if not input_data:
            raise ValueError("Input cannot be empty")
        return True


def test_solution():
    """Test the solution"""
    sol = Solution()
    
    # Test case 1
    test_input_1 = []
    expected_1 = []
    result_1 = sol.solve(test_input_1)
    assert result_1 == expected_1, f"Test 1 failed: expected {{expected_1}}, got {{result_1}}"
    print("✓ Test 1 passed")
    
    print("\\nAll tests passed!")


if __name__ == "__main__":
    test_solution()
'''


# ========================================
# WEB GENERATOR
# ========================================
class WebGenerator:
    """Generate Web application code"""
    
    def generate(self, project_name, description, options):
        framework = options.get('framework', 'Flask')
        
        return {
            'main_code': self._generate_web_code(project_name, description, framework),
            'utils_code': "# Web utilities",
            'requirements': self._get_web_requirements(framework),
            'readme': f"# {project_name}\\n\\n{description}\\n\\nFramework: {framework}"
        }
    
    def _generate_web_code(self, project_name, description, framework):
        if 'Flask' in framework:
            return self._generate_flask_app(project_name)
        elif 'Django' in framework:
            return self._generate_django_app(project_name)
        else:
            return self._generate_flask_app(project_name)
    
    def _generate_flask_app(self, project_name):
        return f'''"""
{project_name} - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    """API endpoint"""
    if request.method == 'POST':
        data = request.get_json()
        # Process data
        return jsonify({{'status': 'success', 'data': data}})
    return jsonify({{'message': 'API is working'}})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''
    
    def _generate_django_app(self, project_name):
        return "# Django application code"
    
    def _get_web_requirements(self, framework):
        if 'Flask' in framework:
            return "flask==2.3.0\\nflask-cors==4.0.0"
        elif 'Django' in framework:
            return "django==4.2.0\\ndjangorestframework==3.14.0"
        return "flask==2.3.0"   