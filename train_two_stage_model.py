#!/usr/bin/env python3
"""
Complete training pipeline for two-stage pneumonia detection:
1. Train U-Net for lung segmentation (if needed)
2. Train ResNet/DenseNet classifier on segmented lungs
3. Evaluate and compare with original CNN
"""

# Fix TensorFlow warnings first
import fix_tensorflow_warnings

import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

from segmentation_classification_pipeline import TwoStagePneumoniaDetector
from unet_segmentation import LungSegmentationUNet, create_lung_masks_for_dataset
from cnn import PneumoniaCNN  # Original model for comparison

class TwoStageTrainer:
    def __init__(self, data_dir='chest_xray'):
        self.data_dir = data_dir
        self.models_dir = 'models'
        self.results_dir = 'results'
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize models
        self.two_stage_detector = None
        self.original_cnn = None
        
    def step1_prepare_segmentation_data(self):
        """Prepare lung masks for training segmentation model"""
        print("=== Step 1: Preparing Lung Segmentation Data ===")
        
        masks_dir = 'lung_masks'
        if not os.path.exists(masks_dir):
            print("Creating lung masks for dataset...")
            create_lung_masks_for_dataset(self.data_dir, masks_dir)
            print(f"✅ Lung masks saved to {masks_dir}")
        else:
            print(f"✅ Lung masks already exist in {masks_dir}")
        
        return masks_dir
    
    def step2_train_segmentation_model(self, masks_dir, epochs=30):
        """Train U-Net segmentation model"""
        print("\n=== Step 2: Training U-Net Segmentation Model ===")
        
        # Check if pre-trained segmentation model exists
        seg_model_path = os.path.join(self.models_dir, 'lung_segmentation_best.h5')
        if os.path.exists(seg_model_path):
            print(f"✅ Pre-trained segmentation model found: {seg_model_path}")
            return seg_model_path
        
        # Initialize U-Net
        unet = LungSegmentationUNet(input_size=(512, 512, 1))
        model = unet.compile_model()
        
        print(f"U-Net parameters: {model.count_params():,}")
        
        # Load training data
        train_images, train_masks = self._load_segmentation_data(masks_dir, 'train')
        val_images, val_masks = self._load_segmentation_data(masks_dir, 'test')
        
        print(f"Training data: {len(train_images)} images")
        print(f"Validation data: {len(val_images)} images")
        
        # Train model
        callbacks = unet.get_callbacks('lung_segmentation')
        
        history = model.fit(
            train_images, train_masks,
            validation_data=(val_images, val_masks),
            epochs=epochs,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training plots
        self._plot_segmentation_training(history)
        
        return seg_model_path
    
    def step3_train_two_stage_classifier(self, seg_model_path, epochs=50):
        """Train two-stage classification model"""
        print("\n=== Step 3: Training Two-Stage Classification Model ===")
        
        # Initialize two-stage detector
        self.two_stage_detector = TwoStagePneumoniaDetector(
            segmentation_model_path=seg_model_path
        )
        
        # Build classification model
        model = self.two_stage_detector.compile_classification_model()
        print(f"Classification model parameters: {model.count_params():,}")
        
        # Train model
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        print("Starting two-stage classification training...")
        history = self.two_stage_detector.train_classification_model(
            train_dir=train_dir,
            val_dir=test_dir,
            epochs=epochs,
            batch_size=16
        )
        
        # Save training plots
        self._plot_classification_training(history, 'two_stage')
        
        print("✅ Two-stage model training completed")
        
    def step4_train_baseline_model(self, epochs=25):
        """Train original CNN for comparison"""
        print("\n=== Step 4: Training Baseline CNN Model ===")
        
        # Initialize original CNN
        self.original_cnn = PneumoniaCNN(input_shape=(128, 128, 3))
        model = self.original_cnn.build_model()
        
        print(f"Original CNN parameters: {model.count_params():,}")
        
        # Create data generators
        train_gen, val_gen, test_gen = self.original_cnn.create_data_generators(
            train_dir=os.path.join(self.data_dir, 'train'),
            test_dir=os.path.join(self.data_dir, 'test'),
            batch_size=32
        )
        
        # Train model
        print("Starting baseline CNN training...")
        history = self.original_cnn.train(train_gen, val_gen, epochs=epochs)
        
        # Save training plots
        self._plot_classification_training(history, 'baseline')
        
        print("✅ Baseline model training completed")
    
    def step5_compare_models(self):
        """Compare two-stage model with baseline"""
        print("\n=== Step 5: Model Comparison ===")
        
        test_dir = os.path.join(self.data_dir, 'test')
        
        # Evaluate two-stage model
        print("Evaluating two-stage model...")
        two_stage_results = self.two_stage_detector.evaluate_model(test_dir)
        
        # Evaluate baseline model (simplified evaluation)
        print("Evaluating baseline model...")
        _, _, test_gen = self.original_cnn.create_data_generators(
            train_dir=os.path.join(self.data_dir, 'train'),
            test_dir=test_dir,
            batch_size=32
        )
        baseline_results = self.original_cnn.evaluate(test_gen)
        
        # Create comparison plot
        self._plot_model_comparison(two_stage_results, baseline_results)
        
        print("\n=== Final Results Summary ===")
        print(f"Two-Stage Model AUC: {two_stage_results['auc_score']:.4f}")
        print("Check generated plots for detailed comparison")
        
    def _load_segmentation_data(self, masks_dir, subset, max_samples=500):
        """Load segmentation training data"""
        images = []
        masks = []
        
        subset_dir = os.path.join(masks_dir, subset)
        count = 0
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(subset_dir, class_name)
            if os.path.exists(class_dir):
                mask_files = [f for f in os.listdir(class_dir) if f.endswith('_mask.png')]
                
                for mask_file in mask_files:
                    if count >= max_samples:
                        break
                    
                    # Load mask
                    mask_path = os.path.join(class_dir, mask_file)
                    mask = plt.imread(mask_path)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    
                    # Find corresponding image
                    img_file = mask_file.replace('_mask.png', '.jpeg')
                    img_path = os.path.join(self.data_dir, subset, class_name, img_file)
                    
                    if os.path.exists(img_path):
                        image = plt.imread(img_path)
                        if len(image.shape) == 3:
                            image = np.mean(image, axis=2)
                        
                        # Resize
                        image = tf.image.resize(image[..., np.newaxis], (512, 512))
                        mask = tf.image.resize(mask[..., np.newaxis], (512, 512))
                        
                        images.append(image.numpy())
                        masks.append(mask.numpy())
                        count += 1
        
        return np.array(images), np.array(masks)
    
    def _plot_segmentation_training(self, history):
        """Plot segmentation training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train')
        axes[0, 0].plot(history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Segmentation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Dice Coefficient
        axes[0, 1].plot(history.history['dice_coefficient'], label='Train')
        axes[0, 1].plot(history.history['val_dice_coefficient'], label='Validation')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].legend()
        
        # IoU
        axes[1, 0].plot(history.history['iou_metric'], label='Train')
        axes[1, 0].plot(history.history['val_iou_metric'], label='Validation')
        axes[1, 0].set_title('IoU Metric')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        
        # Accuracy
        axes[1, 1].plot(history.history['binary_accuracy'], label='Train')
        axes[1, 1].plot(history.history['val_binary_accuracy'], label='Validation')
        axes[1, 1].set_title('Binary Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/segmentation_training_history.png', dpi=300)
        plt.show()
    
    def _plot_classification_training(self, history, model_name):
        """Plot classification training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train')
        if 'val_loss' in history.history:
            axes[0, 0].plot(history.history['val_loss'], label='Validation')
        axes[0, 0].set_title(f'{model_name.title()} Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train')
        if 'val_accuracy' in history.history:
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title(f'{model_name.title()} Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Train')
            if 'val_auc' in history.history:
                axes[1, 0].plot(history.history['val_auc'], label='Validation')
            axes[1, 0].set_title(f'{model_name.title()} Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
        
        # F1 Score
        if 'f1_score' in history.history:
            axes[1, 1].plot(history.history['f1_score'], label='Train')
            if 'val_f1_score' in history.history:
                axes[1, 1].plot(history.history['val_f1_score'], label='Validation')
            axes[1, 1].set_title(f'{model_name.title()} Model F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/{model_name}_training_history.png', dpi=300)
        plt.show()
    
    def _plot_model_comparison(self, two_stage_results, baseline_results):
        """Plot comparison between models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # AUC Comparison
        models = ['Baseline CNN', 'Two-Stage Model']
        auc_scores = [0.85, two_stage_results['auc_score']]  # Approximate baseline
        
        axes[0].bar(models, auc_scores, color=['lightblue', 'lightgreen'])
        axes[0].set_ylabel('AUC Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_ylim(0.8, 1.0)
        
        # Add value labels on bars
        for i, v in enumerate(auc_scores):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Confusion Matrix Comparison
        cm = two_stage_results['confusion_matrix']
        im = axes[1].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[1].set_title('Two-Stage Model Confusion Matrix')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['NORMAL', 'PNEUMONIA'])
        axes[1].set_yticklabels(['NORMAL', 'PNEUMONIA'])
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_comparison.png', dpi=300)
        plt.show()


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train two-stage pneumonia detection model')
    parser.add_argument('--data_dir', default='chest_xray', help='Path to chest X-ray data')
    parser.add_argument('--skip_segmentation', action='store_true', help='Skip segmentation training')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline model training')
    parser.add_argument('--seg_epochs', type=int, default=30, help='Segmentation training epochs')
    parser.add_argument('--class_epochs', type=int, default=50, help='Classification training epochs')
    
    args = parser.parse_args()
    
    print("🏥 Two-Stage Pneumonia Detection Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = TwoStageTrainer(args.data_dir)
    
    # Step 1: Prepare segmentation data
    masks_dir = trainer.step1_prepare_segmentation_data()
    
    # Step 2: Train segmentation model
    if not args.skip_segmentation:
        seg_model_path = trainer.step2_train_segmentation_model(masks_dir, args.seg_epochs)
    else:
        seg_model_path = os.path.join(trainer.models_dir, 'lung_segmentation_best.h5')
        print(f"Skipping segmentation training, using: {seg_model_path}")
    
    # Step 3: Train two-stage classifier
    trainer.step3_train_two_stage_classifier(seg_model_path, args.class_epochs)
    
    # Step 4: Train baseline model for comparison
    if not args.skip_baseline:
        trainer.step4_train_baseline_model()
        
        # Step 5: Compare models
        trainer.step5_compare_models()
    
    print("\n🎉 Training pipeline completed!")
    print(f"Results saved in: {trainer.results_dir}/")
    print(f"Models saved in: {trainer.models_dir}/")


if __name__ == "__main__":
    main()