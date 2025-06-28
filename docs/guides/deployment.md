# Deployment Guide

Complete guide for deploying pneumonia detection models in production environments.

## Overview

This guide covers various deployment strategies for the PneumoniaCNN models:

- **Local Inference**: Single machine deployment
- **REST API Service**: Web service deployment
- **Edge Deployment**: Mobile and edge devices
- **Cloud Deployment**: Scalable cloud solutions
- **Batch Processing**: Large-scale batch inference

## Model Preparation

### Model Export

#### Save Trained Model
```python
from src.models.cnn import PneumoniaCNN

# Load and save model in deployment format
model = PneumoniaCNN(model_path="models/best_model.h5")

# Save as SavedModel (recommended for deployment)
model.model.save("deployment/pneumonia_model", save_format='tf')

# Save as TensorFlow Lite (for mobile/edge)
converter = tf.lite.TFLiteConverter.from_saved_model("deployment/pneumonia_model")
tflite_model = converter.convert()
with open("deployment/pneumonia_model.tflite", "wb") as f:
    f.write(tflite_model)
```

#### ONNX Export (Cross-Platform)
```python
import tf2onnx
import onnx

# Convert to ONNX format
onnx_model, _ = tf2onnx.convert.from_saved_model(
    "deployment/pneumonia_model",
    output_path="deployment/pneumonia_model.onnx"
)

# Verify ONNX model
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

### Model Optimization

#### Quantization for Edge Deployment
```python
# Post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model("deployment/pneumonia_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open("deployment/pneumonia_model_quantized.tflite", "wb") as f:
    f.write(quantized_model)

# Compare model sizes
import os
original_size = os.path.getsize("deployment/pneumonia_model.tflite")
quantized_size = os.path.getsize("deployment/pneumonia_model_quantized.tflite")
print(f"Original: {original_size/1024/1024:.2f} MB")
print(f"Quantized: {quantized_size/1024/1024:.2f} MB")
print(f"Compression: {100*(1-quantized_size/original_size):.1f}%")
```

## Local Inference

### Simple Prediction Script
```python
#!/usr/bin/env python
"""
Simple inference script for local deployment.
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

class PneumoniaPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model."""
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[1:3]  # Get height, width
        
    def preprocess_image(self, image_path):
        """Preprocess image for prediction."""
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.input_shape)
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image_path, threshold=0.5):
        """Make prediction on single image."""
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(processed_image)[0][0]
        
        # Interpret result
        is_pneumonia = prediction > threshold
        confidence = prediction if is_pneumonia else 1 - prediction
        
        return {
            'prediction': 'PNEUMONIA' if is_pneumonia else 'NORMAL',
            'confidence': float(confidence),
            'raw_score': float(prediction)
        }
    
    def predict_batch(self, image_paths, threshold=0.5):
        """Make predictions on multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, threshold)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results

def main():
    parser = argparse.ArgumentParser(description='Pneumonia Detection Inference')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--batch', help='Path to directory of images')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PneumoniaPredictor(args.model)
    
    if args.image:
        # Single image prediction
        result = predictor.predict(args.image, args.threshold)
        print(f"Image: {args.image}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
    elif args.batch:
        # Batch prediction
        import glob
        image_paths = glob.glob(f"{args.batch}/*.jpg") + glob.glob(f"{args.batch}/*.png")
        results = predictor.predict_batch(image_paths, args.threshold)
        
        for result in results:
            if 'error' in result:
                print(f"Error processing {result['image_path']}: {result['error']}")
            else:
                print(f"{result['image_path']}: {result['prediction']} ({result['confidence']:.3f})")

if __name__ == "__main__":
    main()
```

### Usage Examples
```bash
# Single image prediction
python predict.py --model models/best_model.h5 --image data/test_image.jpg

# Batch prediction
python predict.py --model models/best_model.h5 --batch data/test_images/

# Custom threshold
python predict.py --model models/best_model.h5 --image test.jpg --threshold 0.7
```

## REST API Service

### Flask API Implementation
```python
#!/usr/bin/env python
"""
Flask REST API for pneumonia detection service.
"""
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model instance
model = None

def load_model(model_path):
    """Load trained model at startup."""
    global model
    model = tf.keras.models.load_model(model_path)
    app.logger.info(f"Model loaded from {model_path}")

def preprocess_image(image_data, target_size=(128, 128)):
    """Preprocess image for prediction."""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize to model input size
        image = image.resize(target_size)
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get optional parameters
        threshold = float(request.form.get('threshold', 0.5))
        
        # Read and preprocess image
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Interpret result
        is_pneumonia = prediction > threshold
        confidence = prediction if is_pneumonia else 1 - prediction
        
        result = {
            'prediction': 'PNEUMONIA' if is_pneumonia else 'NORMAL',
            'confidence': float(confidence),
            'raw_score': float(prediction),
            'threshold': threshold
        }
        
        app.logger.info(f"Prediction made: {result}")
        return jsonify(result)
        
    except ValueError as e:
        app.logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """Prediction endpoint for base64 encoded images."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception:
            return jsonify({'error': 'Invalid base64 image data'}), 400
        
        threshold = float(data.get('threshold', 0.5))
        
        # Preprocess and predict
        processed_image = preprocess_image(image_data)
        prediction = model.predict(processed_image)[0][0]
        
        # Interpret result
        is_pneumonia = prediction > threshold
        confidence = prediction if is_pneumonia else 1 - prediction
        
        result = {
            'prediction': 'PNEUMONIA' if is_pneumonia else 'NORMAL',
            'confidence': float(confidence),
            'raw_score': float(prediction),
            'threshold': threshold
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Base64 prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pneumonia Detection API Server')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    load_model(args.model)
    
    # Start server
    app.run(host=args.host, port=args.port, debug=args.debug)
```

### API Testing
```bash
# Start the API server
python api_server.py --model models/best_model.h5 --port 5000

# Test health check
curl http://localhost:5000/health

# Test prediction with file upload
curl -X POST \
  -F "image=@test_image.jpg" \
  -F "threshold=0.6" \
  http://localhost:5000/predict

# Test with base64 encoded image
python -c "
import base64
import requests

with open('test_image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post('http://localhost:5000/predict/base64', 
                        json={'image': image_b64, 'threshold': 0.5})
print(response.json())
"
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "api_server.py", "--model", "models/best_model.h5", "--host", "0.0.0.0"]
```

```bash
# Build and run Docker container
docker build -t pneumonia-detection-api .
docker run -p 5000:5000 pneumonia-detection-api

# Or with docker-compose
# docker-compose.yml
version: '3.8'
services:
  pneumonia-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
```

## Edge Deployment

### TensorFlow Lite Inference
```python
#!/usr/bin/env python
"""
TensorFlow Lite inference for edge deployment.
"""
import tensorflow as tf
import numpy as np
from PIL import Image

class TFLitePredictor:
    def __init__(self, model_path):
        """Initialize TFLite predictor."""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]  # height, width
        
    def preprocess_image(self, image_path):
        """Preprocess image for TFLite model."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.input_shape)
        
        # Convert to array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image_path, threshold=0.5):
        """Make prediction using TFLite model."""
        # Preprocess image
        input_data = self.preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction = output_data[0][0]
        
        # Interpret result
        is_pneumonia = prediction > threshold
        confidence = prediction if is_pneumonia else 1 - prediction
        
        return {
            'prediction': 'PNEUMONIA' if is_pneumonia else 'NORMAL',
            'confidence': float(confidence),
            'raw_score': float(prediction)
        }

# Usage example
predictor = TFLitePredictor('deployment/pneumonia_model.tflite')
result = predictor.predict('test_image.jpg')
print(result)
```

### Mobile App Integration (Android/iOS)

#### Android Integration
```java
// Android TensorFlow Lite integration
public class PneumoniaDetector {
    private Interpreter tflite;
    private static final int INPUT_SIZE = 128;
    
    public PneumoniaDetector(Context context) {
        try {
            tflite = new Interpreter(loadModelFile(context, "pneumonia_model.tflite"));
        } catch (IOException e) {
            Log.e("PneumoniaDetector", "Error loading model", e);
        }
    }
    
    public float predict(Bitmap bitmap) {
        // Preprocess bitmap
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        float[][][][] input = new float[1][INPUT_SIZE][INPUT_SIZE][3];
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                int pixel = resized.getPixel(i, j);
                input[0][i][j][0] = (Color.red(pixel) / 255.0f);
                input[0][i][j][1] = (Color.green(pixel) / 255.0f);
                input[0][i][j][2] = (Color.blue(pixel) / 255.0f);
            }
        }
        
        // Run inference
        float[][] output = new float[1][1];
        tflite.run(input, output);
        
        return output[0][0];
    }
}
```

## Cloud Deployment

### AWS SageMaker Deployment
```python
#!/usr/bin/env python
"""
AWS SageMaker deployment script.
"""
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import boto3

def deploy_to_sagemaker(model_path, role_arn, endpoint_name):
    """Deploy model to AWS SageMaker."""
    
    # Upload model to S3
    sagemaker_session = sagemaker.Session()
    model_artifacts = sagemaker_session.upload_data(
        path=model_path,
        bucket=sagemaker_session.default_bucket(),
        key_prefix='pneumonia-detection/model'
    )
    
    # Create TensorFlow model
    tensorflow_model = TensorFlowModel(
        model_data=model_artifacts,
        role=role_arn,
        framework_version='2.8',
        py_version='py39'
    )
    
    # Deploy to endpoint
    predictor = tensorflow_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name
    )
    
    return predictor

# Usage
predictor = deploy_to_sagemaker(
    model_path='deployment/pneumonia_model',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    endpoint_name='pneumonia-detection-endpoint'
)
```

### Google Cloud AI Platform
```python
#!/usr/bin/env python
"""
Google Cloud AI Platform deployment.
"""
from google.cloud import aiplatform

def deploy_to_vertex_ai(model_path, project_id, region, endpoint_name):
    """Deploy model to Google Cloud Vertex AI."""
    
    # Initialize AI Platform
    aiplatform.init(project=project_id, location=region)
    
    # Upload model
    model = aiplatform.Model.upload(
        display_name="pneumonia-detection",
        artifact_uri=model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest"
    )
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    
    # Deploy model to endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3
    )
    
    return endpoint

# Usage
endpoint = deploy_to_vertex_ai(
    model_path="gs://my-bucket/pneumonia-model/",
    project_id="my-project-id",
    region="us-central1",
    endpoint_name="pneumonia-detection-endpoint"
)
```

## Batch Processing

### Large-Scale Batch Inference
```python
#!/usr/bin/env python
"""
Batch processing script for large-scale inference.
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

class BatchProcessor:
    def __init__(self, model_path, batch_size=32, num_workers=4):
        """Initialize batch processor."""
        self.model = tf.keras.models.load_model(model_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def preprocess_batch(self, image_paths):
        """Preprocess a batch of images."""
        batch_images = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                image = tf.keras.preprocessing.image.load_img(
                    image_path, 
                    target_size=(128, 128),
                    color_mode='rgb'
                )
                image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                batch_images.append(image_array)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                
        return np.array(batch_images), valid_paths
    
    def process_batch(self, image_paths, threshold=0.5):
        """Process a batch of images."""
        if not image_paths:
            return []
            
        # Preprocess batch
        batch_images, valid_paths = self.preprocess_batch(image_paths)
        
        if len(batch_images) == 0:
            return []
        
        # Make predictions
        predictions = self.model.predict(batch_images, batch_size=self.batch_size)
        
        # Process results
        results = []
        for i, (path, pred) in enumerate(zip(valid_paths, predictions)):
            is_pneumonia = pred[0] > threshold
            confidence = pred[0] if is_pneumonia else 1 - pred[0]
            
            results.append({
                'image_path': str(path),
                'prediction': 'PNEUMONIA' if is_pneumonia else 'NORMAL',
                'confidence': float(confidence),
                'raw_score': float(pred[0])
            })
            
        return results
    
    def process_directory(self, image_dir, output_file, threshold=0.5):
        """Process all images in a directory."""
        image_dir = Path(image_dir)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f'**/*{ext}'))
            image_paths.extend(image_dir.glob(f'**/*{ext.upper()}'))
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process in batches
        all_results = []
        start_time = time.time()
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = self.process_batch(batch_paths, threshold)
            all_results.extend(batch_results)
            
            # Progress update
            processed = min(i + self.batch_size, len(image_paths))
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"Processed {processed}/{len(image_paths)} images ({rate:.1f} img/sec)")
        
        # Save results
        df = pd.DataFrame(all_results)
        df.to_csv(output_file, index=False)
        
        # Summary statistics
        total_time = time.time() - start_time
        pneumonia_count = len(df[df['prediction'] == 'PNEUMONIA'])
        normal_count = len(df[df['prediction'] == 'NORMAL'])
        
        print(f"\nProcessing complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average rate: {len(image_paths)/total_time:.1f} images/second")
        print(f"Results: {pneumonia_count} PNEUMONIA, {normal_count} NORMAL")
        print(f"Results saved to: {output_file}")
        
        return df

# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Pneumonia Detection')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input-dir', required=True, help='Directory containing images')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    
    args = parser.parse_args()
    
    processor = BatchProcessor(args.model, batch_size=args.batch_size)
    results = processor.process_directory(args.input_dir, args.output, args.threshold)
```

## Monitoring and Maintenance

### Model Performance Monitoring
```python
#!/usr/bin/env python
"""
Production model monitoring.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

class ModelMonitor:
    def __init__(self, db_path="model_monitoring.db"):
        """Initialize monitoring database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize monitoring database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                image_path TEXT,
                prediction TEXT,
                confidence REAL,
                raw_score REAL,
                processing_time REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_name TEXT,
                metric_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, image_path, prediction, confidence, raw_score, processing_time):
        """Log a prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, image_path, prediction, confidence, raw_score, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), image_path, prediction, confidence, raw_score, processing_time))
        
        conn.commit()
        conn.close()
    
    def get_performance_metrics(self, hours=24):
        """Get performance metrics for the last N hours."""
        conn = sqlite3.connect(self.db_path)
        
        # Get predictions from last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        
        conn.close()
        
        if len(df) == 0:
            return {}
        
        # Calculate metrics
        metrics = {
            'total_predictions': len(df),
            'pneumonia_predictions': len(df[df['prediction'] == 'PNEUMONIA']),
            'normal_predictions': len(df[df['prediction'] == 'NORMAL']),
            'avg_confidence': df['confidence'].mean(),
            'avg_processing_time': df['processing_time'].mean(),
            'prediction_rate': len(df) / hours,  # predictions per hour
            'confidence_distribution': df['confidence'].describe().to_dict()
        }
        
        return metrics
    
    def detect_drift(self, window_hours=24, reference_hours=168):
        """Simple drift detection based on prediction distribution."""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions
        recent_cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_df = pd.read_sql_query('''
            SELECT * FROM predictions WHERE timestamp > ?
        ''', conn, params=(recent_cutoff,))
        
        # Get reference predictions (last week)
        reference_cutoff = datetime.now() - timedelta(hours=reference_hours)
        reference_df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(reference_cutoff, recent_cutoff))
        
        conn.close()
        
        if len(recent_df) == 0 or len(reference_df) == 0:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Compare prediction distributions
        recent_pneumonia_rate = len(recent_df[recent_df['prediction'] == 'PNEUMONIA']) / len(recent_df)
        reference_pneumonia_rate = len(reference_df[reference_df['prediction'] == 'PNEUMONIA']) / len(reference_df)
        
        # Simple threshold-based drift detection
        drift_threshold = 0.1  # 10% change in prediction rate
        drift_detected = abs(recent_pneumonia_rate - reference_pneumonia_rate) > drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'recent_pneumonia_rate': recent_pneumonia_rate,
            'reference_pneumonia_rate': reference_pneumonia_rate,
            'change': recent_pneumonia_rate - reference_pneumonia_rate
        }

# Usage in production API
monitor = ModelMonitor()

# Log predictions
start_time = time.time()
result = predictor.predict(image_path)
processing_time = time.time() - start_time

monitor.log_prediction(
    image_path=image_path,
    prediction=result['prediction'],
    confidence=result['confidence'],
    raw_score=result['raw_score'],
    processing_time=processing_time
)

# Check for drift daily
drift_info = monitor.detect_drift()
if drift_info['drift_detected']:
    print(f"Model drift detected: {drift_info}")
```

## Best Practices

### Security Considerations
1. **Input Validation**: Always validate and sanitize image inputs
2. **Rate Limiting**: Implement API rate limiting to prevent abuse
3. **Authentication**: Add proper authentication for production APIs
4. **HTTPS**: Use HTTPS for all API communications
5. **Model Protection**: Protect model files from unauthorized access

### Performance Optimization
1. **Model Optimization**: Use quantization and pruning for edge deployment
2. **Caching**: Implement intelligent caching for repeated requests
3. **Load Balancing**: Use load balancers for high-traffic scenarios
4. **Async Processing**: Implement asynchronous processing for batch jobs
5. **Monitoring**: Continuously monitor performance metrics

### Reliability and Maintenance
1. **Health Checks**: Implement comprehensive health monitoring
2. **Graceful Degradation**: Handle failures gracefully
3. **Model Versioning**: Maintain multiple model versions
4. **Rollback Strategy**: Plan for quick rollbacks if issues arise
5. **Documentation**: Maintain deployment documentation

This deployment guide provides a comprehensive foundation for deploying pneumonia detection models across various environments and scales.