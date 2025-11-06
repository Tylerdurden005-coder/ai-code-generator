// ========================================
// AI CodeGen Pro - Main JavaScript
// Frontend Logic & Interactions
// ========================================

// ========================================
// Global Variables & State Management
// ========================================
const state = {
    currentCategory: 'ml',
    currentTab: 'main',
    generatedCode: {
        main: '',
        utils: '',
        requirements: '',
        readme: ''
    },
    theme: 'light'
};

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// ========================================
// Initialization & Event Listeners
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    attachEventListeners();
    animateHeroCode();
    initParticles();
    checkThemePreference();
});

function initializeApp() {
    console.log('AI CodeGen Pro initialized');
    updateCategoryVisibility('ml');
}

function attachEventListeners() {
    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Mobile Menu Toggle
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', toggleMobileMenu);
    }

    // Smooth Scroll for Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
                updateActiveNavLink(this);
            }
        });
    });

    // Active Navigation on Scroll
    window.addEventListener('scroll', handleScroll);

    // Form Input Validation
    const projectName = document.getElementById('projectName');
    const projectDescription = document.getElementById('projectDescription');
    
    if (projectName) {
        projectName.addEventListener('input', validateProjectName);
    }
    
    if (projectDescription) {
        projectDescription.addEventListener('input', validateDescription);
    }
}

// ========================================
// Theme Management
// ========================================
function checkThemePreference() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    state.theme = savedTheme;
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const newTheme = state.theme === 'light' ? 'dark' : 'light';
    state.theme = newTheme;
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
    showToast(`Switched to ${newTheme} mode`);
}

function updateThemeIcon(theme) {
    const themeIcon = document.querySelector('#themeToggle i');
    if (themeIcon) {
        themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

// ========================================
// Navigation & Scroll Handling
// ========================================
function toggleMobileMenu() {
    const navMenu = document.querySelector('.nav-menu');
    const mobileToggle = document.getElementById('mobileMenuToggle');
    
    if (navMenu && mobileToggle) {
        navMenu.classList.toggle('active');
        mobileToggle.classList.toggle('active');
    }
}

function handleScroll() {
    const header = document.querySelector('.main-header');
    if (header) {
        if (window.scrollY > 100) {
            header.style.boxShadow = 'var(--shadow-lg)';
        } else {
            header.style.boxShadow = 'none';
        }
    }

    // Update active navigation link
    const sections = document.querySelectorAll('section[id]');
    const scrollPosition = window.scrollY + 100;

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);
            if (navLink) {
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.remove('active');
                });
                navLink.classList.add('active');
            }
        }
    });
}

function updateActiveNavLink(clickedLink) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    clickedLink.classList.add('active');
}

function scrollToGenerator() {
    const generatorSection = document.getElementById('generator');
    if (generatorSection) {
        generatorSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// ========================================
// Category Selection & Management
// ========================================
function selectCategory(category) {
    state.currentCategory = category;
    
    // Update button states
    document.querySelectorAll('.category-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const selectedBtn = document.querySelector(`[data-category="${category}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }
    
    // Update form visibility
    updateCategoryVisibility(category);
    
    // Clear previous code
    clearGeneratedCode();
}

function updateCategoryVisibility(category) {
    // Hide all category-specific options
    const optionGroups = ['mlOptions', 'dlOptions', 'nlpOptions', 'rlOptions', 'dsaOptions', 'webOptions'];
    optionGroups.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'none';
        }
    });
    
    // Show selected category options
    const selectedOptions = document.getElementById(`${category}Options`);
    if (selectedOptions) {
        selectedOptions.style.display = 'block';
    }
}

// ========================================
// Code Generation
// ========================================
async function generateCode() {
    // Validate inputs
    const projectName = document.getElementById('projectName').value.trim();
    const projectDescription = document.getElementById('projectDescription').value.trim();
    
    if (!projectName) {
        showToast('Please enter a project name', 'error');
        return;
    }
    
    if (!projectDescription) {
        showToast('Please provide a project description', 'error');
        return;
    }
    
    // Show loading overlay
    showLoading();
    
    // Prepare request data
    const requestData = prepareGenerationRequest(projectName, projectDescription);
    
    try {
        // Call backend API
        const response = await fetch(`${API_BASE_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate code');
        }
        
        const result = await response.json();
        
        // Store generated code
        state.generatedCode = {
            main: result.main_code || '',
            utils: result.utils_code || '',
            requirements: result.requirements || '',
            readme: result.readme || ''
        };
        
        // Display generated code
        displayGeneratedCode();
        
        hideLoading();
        showToast('Code generated successfully!', 'success');
        
    } catch (error) {
        console.error('Generation error:', error);
        hideLoading();
        
        // Fallback to mock generation for demo
        generateMockCode(projectName, projectDescription);
        showToast('Using demo mode - Backend not connected', 'warning');
    }
}

function prepareGenerationRequest(projectName, projectDescription) {
    const category = state.currentCategory;
    const data = {
        project_name: projectName,
        description: projectDescription,
        category: category,
        include_comments: document.getElementById('includeComments')?.checked || false,
        include_tests: document.getElementById('includeTests')?.checked || false,
        include_documentation: document.getElementById('includeDocumentation')?.checked || false,
        include_visualization: document.getElementById('includeVisualization')?.checked || false
    };
    
    // Add category-specific parameters
    switch(category) {
        case 'ml':
            const mlModel = document.querySelector('input[name="mlModel"]:checked');
            data.model_type = mlModel ? mlModel.value : 'classification';
            break;
        case 'dl':
            const dlSelect = document.querySelector('#dlOptions select');
            data.architecture = dlSelect ? dlSelect.value : 'CNN';
            break;
        case 'nlp':
            const nlpSelect = document.querySelector('#nlpOptions select');
            data.task = nlpSelect ? nlpSelect.value : 'Sentiment Analysis';
            break;
        case 'rl':
            const rlSelect = document.querySelector('#rlOptions select');
            data.algorithm = rlSelect ? rlSelect.value : 'Q-Learning';
            break;
        case 'dsa':
            const dsaSelect = document.querySelector('#dsaOptions select');
            data.problem_type = dsaSelect ? dsaSelect.value : 'Arrays';
            break;
        case 'web':
            const webSelect = document.querySelector('#webOptions select');
            data.framework = webSelect ? webSelect.value : 'React + Node.js';
            break;
    }
    
    return data;
}

function generateMockCode(projectName, projectDescription) {
    const category = state.currentCategory;
    
    switch(category) {
        case 'ml':
            state.generatedCode = generateMLCode(projectName, projectDescription);
            break;
        case 'dl':
            state.generatedCode = generateDLCode(projectName, projectDescription);
            break;
        case 'nlp':
            state.generatedCode = generateNLPCode(projectName, projectDescription);
            break;
        case 'rl':
            state.generatedCode = generateRLCode(projectName, projectDescription);
            break;
        case 'dsa':
            state.generatedCode = generateDSACode(projectName, projectDescription);
            break;
        case 'web':
            state.generatedCode = generateWebCode(projectName, projectDescription);
            break;
    }
    
    displayGeneratedCode();
}

// ========================================
// Mock Code Generation Templates
// ========================================
function generateMLCode(projectName, description) {
    const mainCode = `"""
${projectName}
${description}

Author: AI CodeGen Pro
Generated: ${new Date().toLocaleDateString()}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ${projectName.replace(/\s+/g, '')}Model:
    def __init__(self, random_state=42):
        """Initialize the ML model"""
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def load_data(self, file_path):
        """Load and prepare dataset"""
        print(f"Loading data from {file_path}...")
        self.data = pd.read_csv(file_path)
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self, target_column='target'):
        """Preprocess and split data"""
        print("Preprocessing data...")
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train_scaled.shape}")
        print(f"Test set size: {self.X_test_scaled.shape}")
    
    def train(self, cv_folds=5):
        """Train the model with cross-validation"""
        print("Training model...")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, cv=cv_folds
        )
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_trained = True
        print("Training completed!")
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.is_trained:
            raise Exception("Model must be trained before evaluation")
        
        print("\\nEvaluating model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\\nAccuracy: {accuracy:.4f}")
        
        print("\\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        # Feature Importance
        self.plot_feature_importance()
        
        return accuracy
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved to confusion_matrix.png")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved to feature_importance.png")
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained:
            raise Exception("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def save_model(self, filepath='model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise Exception("Model must be trained before saving")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.pkl'):
        """Load trained model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def main():
    """Main execution function"""
    print("="*50)
    print(f"${projectName}")
    print("="*50)
    
    # Initialize model
    model = ${projectName.replace(/\s+/g, '')}Model()
    
    # Load and preprocess data
    model.load_data('data.csv')
    model.preprocess_data(target_column='target')
    
    # Train model
    model.train(cv_folds=5)
    
    # Evaluate model
    accuracy = model.evaluate()
    
    # Save model
    model.save_model('trained_model.pkl')
    
    print("\\nPipeline completed successfully!")

if __name__ == "__main__":
    main()`;

    const utilsCode = `"""
Utility functions for ${projectName}
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def load_and_clean_data(filepath, missing_strategy='mean'):
    """Load and clean data with various strategies"""
    df = pd.read_csv(filepath)
    
    if missing_strategy == 'mean':
        df = df.fillna(df.mean())
    elif missing_strategy == 'median':
        df = df.fillna(df.median())
    elif missing_strategy == 'drop':
        df = df.dropna()
    
    return df

def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics`;

    const requirements = `# ${projectName} Requirements
# Generated by AI CodeGen Pro

numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2`;

    const readme = `# ${projectName}

${description}

## Overview
This project was generated using AI CodeGen Pro and includes a complete machine learning pipeline.

## Features
- Data loading and preprocessing
- Model training with cross-validation
- Performance evaluation and visualization
- Model persistence (save/load)
- Feature importance analysis

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`python
python main.py
\`\`\`

## Project Structure
- \`main.py\`: Main model implementation
- \`utils.py\`: Utility functions
- \`requirements.txt\`: Python dependencies

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies

## Generated by
AI CodeGen Pro - ${new Date().toLocaleDateString()}`;

    return { main: mainCode, utils: utilsCode, requirements, readme };
}

function generateDLCode(projectName, description) {
    const mainCode = `"""
${projectName} - Deep Learning Model
${description}
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class ${projectName.replace(/\s+/g, '')}CNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.history = None
    
    def build_model(self):
        """Build CNN architecture"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        results = self.model.evaluate(X_test, y_test)
        return results
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

def main():
    print("Building ${projectName}...")
    model = ${projectName.replace(/\s+/g, '')}CNN()
    print(model.model.summary())

if __name__ == "__main__":
    main()`;

    return {
        main: mainCode,
        utils: '# Deep Learning utilities',
        requirements: 'tensorflow==2.13.0\nnumpy==1.24.3\nmatplotlib==3.7.2',
        readme: `# ${projectName}\n\nDeep Learning project generated by AI CodeGen Pro`
    };
}

function generateNLPCode(projectName, description) {
    const mainCode = `"""
${projectName} - NLP Model
${description}
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np

class ${projectName.replace(/\s+/g, '')}NLP:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def preprocess(self, texts):
        """Tokenize texts"""
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    def train(self, train_dataset, epochs=3):
        """Train the model"""
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

def main():
    print("Initializing ${projectName}...")
    nlp_model = ${projectName.replace(/\s+/g, '')}NLP()

if __name__ == "__main__":
    main()`;

    return {
        main: mainCode,
        utils: '# NLP utilities',
        requirements: 'transformers==4.30.0\ntorch==2.0.1\nnumpy==1.24.3',
        readme: `# ${projectName}\n\nNLP project generated by AI CodeGen Pro`
    };
}

function generateRLCode(projectName, description) {
    return {
        main: '# Reinforcement Learning code template',
        utils: '# RL utilities',
        requirements: 'gym==0.26.0\nstable-baselines3==2.0.0',
        readme: `# ${projectName}\n\nRL project`
    };
}

function generateDSACode(projectName, description) {
    return {
        main: '# DSA solution template',
        utils: '# DSA utilities',
        requirements: '# No special requirements',
        readme: `# ${projectName}\n\nDSA solution`
    };
}

function generateWebCode(projectName, description) {
    return {
        main: '# Web application backend',
        utils: '# Web utilities',
        requirements: 'flask==2.3.0\ndjango==4.2.0',
        readme: `# ${projectName}\n\nWeb application`
    };
}

// ========================================
// Code Display & Tab Management
// ========================================
function displayGeneratedCode() {
    const codeOutput = document.getElementById('codeOutput');
    if (!codeOutput) return;
    
    // Display the current tab's code
    const currentCode = state.generatedCode[state.currentTab];
    
    if (currentCode) {
        codeOutput.innerHTML = `<pre><code>${escapeHtml(currentCode)}</code></pre>`;
    } else {
        codeOutput.innerHTML = '<div class="placeholder-content"><i class="fas fa-code fa-3x"></i><p>No code generated for this file</p></div>';
    }
}

function switchTab(tabName) {
    state.currentTab = tabName;
    
    // Update tab button states
    document.querySelectorAll('.code-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    const activeTab = document.querySelector(`[onclick="switchTab('${tabName}')"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
    
    // Display code for selected tab
    displayGeneratedCode();
}

function clearGeneratedCode() {
    state.generatedCode = {
        main: '',
        utils: '',
        requirements: '',
        readme: ''
    };
    
    const codeOutput = document.getElementById('codeOutput');
    if (codeOutput) {
        codeOutput.innerHTML = `
            <div class="placeholder-content">
                <i class="fas fa-code fa-3x"></i>
                <p>Your generated code will appear here</p>
                <p class="placeholder-hint">Select a category and describe your project to get started</p>
            </div>
        `;
    }
}

// ========================================
// Code Actions (Copy, Download, Share)
// ========================================
function copyCode() {
    const currentCode = state.generatedCode[state.currentTab];
    
    if (!currentCode) {
        showToast('No code to copy', 'error');
        return;
    }
    
    navigator.clipboard.writeText(currentCode).then(() => {
        showToast('Code copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Failed to copy code', 'error');
    });
}

function downloadCode() {
    const currentCode = state.generatedCode[state.currentTab];
    
    if (!currentCode) {
        showToast('No code to download', 'error');
        return;
    }
    
    const fileExtensions = {
        main: '.py',
        utils: '.py',
        requirements: '.txt',
        readme: '.md'
    };
    
    const filename = state.currentTab + (fileExtensions[state.currentTab] || '.txt');
    const blob = new Blob([currentCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast(`Downloaded ${filename}`, 'success');
}

function shareCode() {
    const projectName = document.getElementById('projectName').value.trim();
    
    if (!projectName) {
        showToast('Please enter a project name first', 'error');
        return;
    }
    
    const shareUrl = `${window.location.origin}?project=${encodeURIComponent(projectName)}`;
    
    navigator.clipboard.writeText(shareUrl).then(() => {
        showToast('Share link copied!', 'success');
    }).catch(err => {
        showToast('Failed to create share link', 'error');
    });
}

// ========================================
// Template Loading
// ========================================
function loadTemplate(templateName) {
    const templates = {
        'image-classifier': {
            name: 'Image Classifier',
            description: 'Build a CNN-based image classification model using TensorFlow/Keras',
            category: 'dl'
        },
        'sentiment-analysis': {
            name: 'Sentiment Analyzer',
            description: 'Create a sentiment analysis model using BERT transformers',
            category: 'nlp'
        },
        'recommendation': {
            name: 'Recommendation System',
            description: 'Build a collaborative filtering recommendation system',
            category: 'ml'
        },
        'chatbot': {
            name: 'AI Chatbot',
            description: 'Create an intelligent chatbot using NLP and transformers',
            category: 'nlp'
        },
        'object-detection': {
            name: 'Object Detector',
            description: 'Implement YOLO-based real-time object detection',
            category: 'dl'
        },
        'time-series': {
            name: 'Time Series Forecaster',
            description: 'Build LSTM model for time series forecasting',
            category: 'dl'
        }
    };
    
    const template = templates[templateName];
    if (!template) return;
    
    // Set form values
    document.getElementById('projectName').value = template.name;
    document.getElementById('projectDescription').value = template.description;
    
    // Switch to appropriate category
    selectCategory(template.category);
    
    // Scroll to generator
    scrollToGenerator();
    
    showToast(`Template "${template.name}" loaded`, 'success');
}

// ========================================
// Form Validation
// ========================================
function validateProjectName(e) {
    const input = e.target;
    const value = input.value.trim();
    
    if (value.length < 3) {
        input.style.borderColor = 'var(--secondary-color)';
    } else {
        input.style.borderColor = 'var(--accent-color)';
    }
}

function validateDescription(e) {
    const input = e.target;
    const value = input.value.trim();
    
    if (value.length < 10) {
        input.style.borderColor = 'var(--secondary-color)';
    } else {
        input.style.borderColor = 'var(--accent-color)';
    }
}

// ========================================
// Loading Overlay
// ========================================
function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('active');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

// ========================================
// Toast Notifications
// ========================================
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    if (!toast || !toastMessage) return;
    
    toastMessage.textContent = message;
    
    // Update icon based on type
    const icon = toast.querySelector('i');
    if (icon) {
        icon.className = type === 'success' ? 'fas fa-check-circle' :
                        type === 'error' ? 'fas fa-exclamation-circle' :
                        type === 'warning' ? 'fas fa-exclamation-triangle' :
                        'fas fa-info-circle';
    }
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ========================================
// Hero Code Animation
// ========================================
function animateHeroCode() {
    const heroCode = document.getElementById('heroCode');
    if (!heroCode) return;
    
    const codeSnippet = `import tensorflow as tf
from tensorflow import keras

# Build Neural Network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2
)

print("Model trained successfully!")`;
    
    let index = 0;
    const speed = 30;
    
    function typeWriter() {
        if (index < codeSnippet.length) {
            heroCode.textContent += codeSnippet.charAt(index);
            index++;
            setTimeout(typeWriter, speed);
        }
    }
    
    typeWriter();
}

// ========================================
// Particle Animation
// ========================================
function initParticles() {
    const particleContainer = document.getElementById('particles');
    if (!particleContainer) return;
    
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.style.position = 'absolute';
        particle.style.width = '2px';
        particle.style.height = '2px';
        particle.style.background = 'rgba(99, 102, 241, 0.5)';
        particle.style.borderRadius = '50%';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animation = `float ${10 + Math.random() * 20}s ease-in-out infinite`;
        particle.style.animationDelay = Math.random() * 5 + 's';
        
        particleContainer.appendChild(particle);
    }
}

// ========================================
// Utility Functions
// ========================================
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ========================================
// API Communication Functions
// ========================================
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        return response.ok;
    } catch (error) {
        return false;
    }
}

async function getAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to fetch models:', error);
    }
    return [];
}

// ========================================
// Keyboard Shortcuts
// ========================================
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to generate code
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generateCode();
    }
    
    // Ctrl/Cmd + S to download code
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        downloadCode();
    }
    
    // Ctrl/Cmd + C when focused on output to copy
    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.closest('.code-output')) {
            e.preventDefault();
            copyCode();
        }
    }
});

// ========================================
// Export Functions for Global Access
// ========================================
window.selectCategory = selectCategory;
window.generateCode = generateCode;
window.switchTab = switchTab;
window.copyCode = copyCode;
window.downloadCode = downloadCode;
window.shareCode = shareCode;
window.loadTemplate = loadTemplate;
window.scrollToGenerator = scrollToGenerator;

console.log('AI CodeGen Pro v1.0 - Frontend loaded successfully!');  