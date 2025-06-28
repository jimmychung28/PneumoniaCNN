# Basic Usage Examples

Practical examples for common use cases with the PneumoniaCNN project.

## Quick Start Examples

### Example 1: Train a Basic Model

```python
#!/usr/bin/env python
"""
Example 1: Basic model training with default settings.
"""
from src.models.cnn import PneumoniaCNN
from src.config.config_loader import ConfigLoader

# Load default configuration
loader = ConfigLoader()
config = loader.load_config("configs/default.yaml")

# Create and train model
model = PneumoniaCNN(config=config, mode='standard')
history = model.train()

# Evaluate model
results = model.evaluate()
print(f"Test Accuracy: {results['accuracy']:.3f}")
print(f"Test AUC: {results['auc']:.3f}")

# Save model
model_path = model.save_model()
print(f"Model saved to: {model_path}")
```

### Example 2: Custom Configuration

```python
#!/usr/bin/env python
"""
Example 2: Training with custom hyperparameters.
"""
from src.config.config_loader import ConfigLoader
from src.models.cnn import PneumoniaCNN

# Create custom configuration
loader = ConfigLoader()
base_config = loader.load_config("configs/default.yaml")

# Override specific parameters
custom_config = {
    "experiment_name": "custom_lr_experiment",
    "model": {
        "learning_rate": 0.01,
        "dropout_rate": 0.3
    },
    "training": {
        "batch_size": 64,
        "epochs": 30
    },
    "data": {
        "image_size": [224, 224],
        "augmentation": {
            "rotation_range": 30,
            "zoom_range": 0.3
        }
    }
}

# Merge configurations
merged_config = loader.merge_configs(base_config.__dict__, custom_config)
config = loader.load_from_dict(merged_config)

# Train model
model = PneumoniaCNN(config=config)
history = model.train()
```

### Example 3: Loading and Using a Trained Model

```python
#!/usr/bin/env python
"""
Example 3: Loading a pre-trained model for inference.
"""
from src.models.cnn import PneumoniaCNN
import numpy as np
from PIL import Image

# Load trained model
model = PneumoniaCNN(model_path="models/best_model.h5")

# Load and preprocess an image
image = Image.open("test_image.jpg").convert('RGB')
image = image.resize((128, 128))
image_array = np.array(image) / 255.0
image_batch = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_batch)
probability = prediction[0][0]

# Interpret result
if probability > 0.5:
    diagnosis = "PNEUMONIA"
    confidence = probability
else:
    diagnosis = "NORMAL"
    confidence = 1 - probability

print(f"Diagnosis: {diagnosis}")
print(f"Confidence: {confidence:.3f}")
print(f"Raw probability: {probability:.3f}")
```

## Configuration Examples

### Example 4: High-Performance Training

```python
#!/usr/bin/env python
"""
Example 4: High-performance training for production.
"""
from src.config.config_loader import ConfigLoader
from src.models.cnn import PneumoniaCNN

# Load high-performance configuration
config = ConfigLoader().load_config("configs/high_performance.yaml")

# Modify for your setup
config.training.batch_size = 128  # Larger batch for better GPU utilization
config.hardware.mixed_precision = True  # Enable mixed precision
config.data.cache = True  # Enable data caching
config.data.prefetch = True  # Enable data prefetching

# Train with optimizations
model = PneumoniaCNN(config=config, mode='high_performance')
history = model.train()

# Monitor performance
print(f"Training completed in {history.history}")
```

### Example 5: Quick Development Iteration

```python
#!/usr/bin/env python
"""
Example 5: Fast iteration for development and debugging.
"""
from src.config.config_loader import ConfigLoader
from src.models.cnn import PneumoniaCNN

# Create quick development config
quick_config = {
    "experiment_name": "dev_quick",
    "training": {
        "epochs": 2,
        "batch_size": 16,
        "validation_split": 0.1
    },
    "data": {
        "augmentation": {"enabled": False},  # Disable augmentation for speed
        "image_size": [64, 64]  # Smaller images for faster processing
    },
    "logging": {
        "tensorboard": {"enabled": False}  # Disable logging for speed
    }
}

loader = ConfigLoader()
config = loader.load_from_dict(quick_config)

# Quick training run
model = PneumoniaCNN(config=config, mode='basic')
history = model.train()
print("Quick development run completed!")
```

## Data Pipeline Examples

### Example 6: Custom Data Loading

```python
#!/usr/bin/env python
"""
Example 6: Custom data loading and preprocessing.
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from src.training.data_pipeline import create_data_pipeline

def load_custom_dataset(data_dir, image_size=(128, 128), batch_size=32):
    """Custom data loading function."""
    
    # Create data pipeline with custom settings
    dataset = create_data_pipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        augment=True,
        shuffle=True,
        cache=True,
        prefetch=True
    )
    
    return dataset

def custom_preprocessing(image, label):
    """Custom preprocessing function."""
    # Apply custom normalization
    image = tf.cast(image, tf.float32) / 255.0
    
    # Apply histogram equalization (simplified)
    image = tf.image.adjust_contrast(image, 1.2)
    
    # Add noise for robustness
    noise = tf.random.normal(tf.shape(image), stddev=0.01)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    return image, label

# Load and preprocess data
train_dataset = load_custom_dataset("data/chest_xray/train")
train_dataset = train_dataset.map(custom_preprocessing)

# Use with model training
from src.models.cnn import PneumoniaCNN
model = PneumoniaCNN()
# Note: You would need to modify the train method to accept custom datasets
```

### Example 7: Data Augmentation Experimentation

```python
#!/usr/bin/env python
"""
Example 7: Experimenting with different augmentation strategies.
"""
import tensorflow as tf
from src.training.data_pipeline import create_data_pipeline

def create_augmentation_pipeline(severity='mild'):
    """Create different augmentation pipelines."""
    
    if severity == 'mild':
        augmentation_config = {
            "enabled": True,
            "rotation_range": 10,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.05,
            "horizontal_flip": True,
            "brightness_range": [0.9, 1.1]
        }
    elif severity == 'moderate':
        augmentation_config = {
            "enabled": True,
            "rotation_range": 20,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "zoom_range": 0.15,
            "horizontal_flip": True,
            "brightness_range": [0.8, 1.2],
            "contrast_range": [0.8, 1.2]
        }
    elif severity == 'aggressive':
        augmentation_config = {
            "enabled": True,
            "rotation_range": 30,
            "width_shift_range": 0.3,
            "height_shift_range": 0.3,
            "zoom_range": 0.25,
            "horizontal_flip": True,
            "vertical_flip": False,
            "brightness_range": [0.7, 1.3],
            "contrast_range": [0.7, 1.3]
        }
    
    return augmentation_config

# Test different augmentation levels
from src.config.config_loader import ConfigLoader
from src.models.cnn import PneumoniaCNN

for severity in ['mild', 'moderate', 'aggressive']:
    print(f"\nTesting {severity} augmentation...")
    
    # Create config with specific augmentation
    config_dict = {
        "experiment_name": f"augmentation_{severity}",
        "training": {"epochs": 5},  # Quick test
        "data": {"augmentation": create_augmentation_pipeline(severity)}
    }
    
    loader = ConfigLoader()
    config = loader.load_from_dict(config_dict)
    
    # Train and evaluate
    model = PneumoniaCNN(config=config)
    history = model.train()
    results = model.evaluate()
    
    print(f"{severity.capitalize()} augmentation results:")
    print(f"  Final training accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
    print(f"  Test accuracy: {results['accuracy']:.3f}")
```

## Model Comparison Examples

### Example 8: Comparing Different Architectures

```python
#!/usr/bin/env python
"""
Example 8: Comparing different model configurations.
"""
from src.config.config_loader import ConfigLoader
from src.models.cnn import PneumoniaCNN
import pandas as pd

def compare_models():
    """Compare different model configurations."""
    
    # Define different configurations to test
    configs = {
        "basic_cnn": {
            "experiment_name": "basic_cnn",
            "model": {
                "dropout_rate": 0.5,
                "learning_rate": 0.001
            },
            "training": {"epochs": 10}
        },
        "high_dropout": {
            "experiment_name": "high_dropout",
            "model": {
                "dropout_rate": 0.7,
                "learning_rate": 0.001
            },
            "training": {"epochs": 10}
        },
        "low_lr": {
            "experiment_name": "low_lr",
            "model": {
                "dropout_rate": 0.5,
                "learning_rate": 0.0001
            },
            "training": {"epochs": 10}
        },
        "large_input": {
            "experiment_name": "large_input",
            "model": {
                "input_shape": [224, 224, 3],
                "dropout_rate": 0.5,
                "learning_rate": 0.001
            },
            "data": {
                "image_size": [224, 224]
            },
            "training": {"epochs": 10}
        }
    }
    
    results = []
    loader = ConfigLoader()
    
    for name, config_dict in configs.items():
        print(f"\nTraining {name}...")
        
        # Load and merge with base config
        base_config = loader.load_config("configs/default.yaml")
        merged_config = loader.merge_configs(base_config.__dict__, config_dict)
        config = loader.load_from_dict(merged_config)
        
        # Train model
        model = PneumoniaCNN(config=config)
        history = model.train()
        eval_results = model.evaluate()
        
        # Store results
        result = {
            'model': name,
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1],
            'test_acc': eval_results['accuracy'],
            'test_auc': eval_results['auc'],
            'best_val_acc': max(history.history['val_accuracy'])
        }
        results.append(result)
        
        print(f"  Test accuracy: {result['test_acc']:.3f}")
        print(f"  Test AUC: {result['test_auc']:.3f}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv("model_comparison_results.csv", index=False)
    
    return df

# Run comparison
results_df = compare_models()
```

## Evaluation and Analysis Examples

### Example 9: Detailed Model Evaluation

```python
#!/usr/bin/env python
"""
Example 9: Comprehensive model evaluation and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.models.cnn import PneumoniaCNN
import seaborn as sns

def detailed_evaluation(model_path, test_data_dir):
    """Perform detailed model evaluation."""
    
    # Load model
    model = PneumoniaCNN(model_path=model_path)
    
    # Load test data
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        batch_size=32,
        image_size=(128, 128),
        label_mode='binary',
        shuffle=False
    )
    
    # Get predictions and true labels
    predictions = model.model.predict(test_dataset)
    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, 
                               target_names=['NORMAL', 'PNEUMONIA']))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prediction distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(predictions[true_labels == 0], bins=30, alpha=0.7, 
             label='NORMAL', color='blue', density=True)
    plt.hist(predictions[true_labels == 1], bins=30, alpha=0.7, 
             label='PNEUMONIA', color='red', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    confidence_scores = np.maximum(predictions, 1 - predictions).flatten()
    plt.hist(confidence_scores, bins=30, alpha=0.7, color='green', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Prediction Confidence Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # Recall for PNEUMONIA
    specificity = tn / (tn + fp)  # Recall for NORMAL
    ppv = tp / (tp + fp)  # Precision for PNEUMONIA
    npv = tn / (tn + fn)  # Precision for NORMAL
    
    print(f"\nDetailed Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Positive Predictive Value: {ppv:.3f}")
    print(f"Negative Predictive Value: {npv:.3f}")
    print(f"AUC-ROC: {roc_auc:.3f}")
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'auc': roc_auc,
        'predictions': predictions,
        'true_labels': true_labels
    }

# Run detailed evaluation
results = detailed_evaluation("models/best_model.h5", "data/chest_xray/test")
```

### Example 10: Error Analysis

```python
#!/usr/bin/env python
"""
Example 10: Analyzing model errors and failure cases.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from src.models.cnn import PneumoniaCNN

def analyze_errors(model_path, test_data_dir, output_dir="error_analysis"):
    """Analyze model errors and visualize failure cases."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    model = PneumoniaCNN(model_path=model_path)
    
    # Get all test images and their paths
    test_path = Path(test_data_dir)
    normal_images = list((test_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((test_path / "PNEUMONIA").glob("*.jpeg"))
    
    all_images = [(path, 0) for path in normal_images] + [(path, 1) for path in pneumonia_images]
    
    # Make predictions on all images
    errors = []
    for image_path, true_label in all_images:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        prediction = model.predict(image_batch)[0][0]
        predicted_label = int(prediction > 0.5)
        
        # Check for errors
        if predicted_label != true_label:
            error_type = "False Positive" if predicted_label == 1 else "False Negative"
            errors.append({
                'image_path': str(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'prediction_score': prediction,
                'error_type': error_type,
                'image_array': image_array
            })
    
    print(f"Found {len(errors)} errors out of {len(all_images)} images")
    print(f"Error rate: {len(errors)/len(all_images)*100:.2f}%")
    
    # Separate error types
    false_positives = [e for e in errors if e['error_type'] == 'False Positive']
    false_negatives = [e for e in errors if e['error_type'] == 'False Negative']
    
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    
    # Visualize worst errors
    def visualize_errors(error_list, title, filename):
        if not error_list:
            return
            
        # Sort by confidence (how wrong the model was)
        error_list.sort(key=lambda x: abs(x['prediction_score'] - x['true_label']), reverse=True)
        
        # Show top 9 worst errors
        n_show = min(9, len(error_list))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'{title} - Worst {n_show} Cases', fontsize=16)
        
        for i in range(n_show):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            error = error_list[i]
            ax.imshow(error['image_array'])
            ax.set_title(f"True: {'PNEUMONIA' if error['true_label'] else 'NORMAL'}\n"
                        f"Pred: {error['prediction_score']:.3f}\n"
                        f"File: {Path(error['image_path']).name}", fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_show, 9):
            row, col = i // 3, i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualize false positives and false negatives
    visualize_errors(false_positives, "False Positives (Normal predicted as Pneumonia)", 
                    "false_positives.png")
    visualize_errors(false_negatives, "False Negatives (Pneumonia predicted as Normal)", 
                    "false_negatives.png")
    
    # Analyze prediction score distributions for errors
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if false_positives:
        fp_scores = [e['prediction_score'] for e in false_positives]
        plt.hist(fp_scores, bins=20, alpha=0.7, color='red', label='False Positives')
        plt.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.title('False Positive Score Distribution')
        plt.legend()
    
    plt.subplot(1, 2, 2)
    if false_negatives:
        fn_scores = [e['prediction_score'] for e in false_negatives]
        plt.hist(fn_scores, bins=20, alpha=0.7, color='blue', label='False Negatives')
        plt.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.title('False Negative Score Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_score_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors

# Run error analysis
errors = analyze_errors("models/best_model.h5", "data/chest_xray/test")
```

These examples provide a comprehensive foundation for using the PneumoniaCNN project across various scenarios, from basic training to advanced analysis and deployment.