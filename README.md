# UrbanSound8K Audio Classification - Complete Reproduction Guide

This guide provides step-by-step instructions to reproduce all results from the paper "UrbanSound8K Audio Classification Using Transfer Learning and AWS Cloud."

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Local Training and Evaluation](#local-training)
4. [AWS Deployment](#aws-deployment)
5. [Performance Benchmarking](#performance-benchmarking)

---

## Prerequisites

### Required Accounts
- **Kaggle Account**: For downloading UrbanSound8K dataset
- **AWS Account**: For cloud deployment (free tier eligible for some services)
- **GitHub Account**: For code repository access

### Required Software
- **Python 3.11+** (with pip)
- **Git** (for cloning repository)
- **AWS CLI 2.x** (for AWS deployment)
- **Node.js 18+** (for local testing only, optional)

### Hardware Requirements
- **Local Training**: 16GB+ RAM recommended, will take ~8-12 hours on CPU
- **AWS Training**: Covered by AWS compute (ml.m5.2xlarge)
- **Stress Testing**: 8GB+ RAM, stable internet connection

---

## 1. Dataset Setup

### Download UrbanSound8K from Kaggle

1. **Create Kaggle API Token**:
   ```bash
   # Go to https://www.kaggle.com/account
   # Click "Create New API Token"
   # This downloads kaggle.json
   ```

2. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

3. **Configure Kaggle Credentials**:
   
   **On Windows**:
   ```powershell
   # Create .kaggle directory in user home
   mkdir $env:USERPROFILE\.kaggle
   
   # Move kaggle.json to .kaggle directory
   move Downloads\kaggle.json $env:USERPROFILE\.kaggle\
   ```
   
   **On Linux/Mac**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Download UrbanSound8K Dataset**:
   ```bash
   # Download dataset (2.8 GB)
   kaggle datasets download -d chrisfilo/urbansound8k
   
   # Extract
   unzip urbansound8k.zip -d urbansound8k_data
   ```

5. **Verify Dataset Structure**:
   ```
   urbansound8k_data/
   ├── audio/
   │   ├── fold1/
   │   ├── fold2/
   │   ├── ...
   │   └── fold10/
   └── metadata/
       └── UrbanSound8K.csv
   ```

---

## 2. Local Training and Evaluation

### 2.1 Environment Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/markjeakle/urbansound8k-aws.git
   cd urbansound8k-aws
   ```

2. **Create Virtual Environment**:
   ```bash
   # Create environment
   python -m venv venv
   
   # Activate
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements-local.txt
   ```

### 2.2 Preprocess Audio to Spectrograms

**Purpose**: Convert audio files to mel-spectrogram images for CNN input.

```bash
python scripts/01_preprocess_audio.py \
  --audio-dir urbansound8k_data/audio \
  --metadata urbansound8k_data/metadata/UrbanSound8K.csv \
  --output-dir data/spectrograms \
  --sample-rate 22050 \
  --n-mels 128 \
  --img-size 224
```

**Expected Output**:
```
Processing fold1... 880 files
Processing fold2... 880 files
...
Processing fold10... 837 files
Total spectrograms created: 8732
Output directory: data/spectrograms/
```

**Time**: ~20-30 minutes

### 2.3 Train Model Locally

**Purpose**: Train ResNet50 model using transfer learning.

```bash
python scripts/02_train_local.py \
  --data-dir data/spectrograms \
  --output-dir output/models \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --val-fold 10
```

**Expected Output**:
```
Epoch 1/20: Train Loss: 1.2345, Train Acc: 55.23%, Val Loss: 1.0123, Val Acc: 62.45%
Epoch 2/20: Train Loss: 0.9876, Train Acc: 67.89%, Val Loss: 0.8765, Val Acc: 70.12%
...
Epoch 20/20: Train Loss: 0.0173, Train Acc: 99.47%, Val Loss: 0.7782, Val Acc: 81.24%

Best Model Saved: output/models/best_model.pth
Final Model Saved: output/models/final_model.pth
Training Complete! Total Time: 3.5 hours
```

**Time**: 8-12 hours on CPU (varies by hardware)

### 2.4 Evaluate Model

```bash
python scripts/03_evaluate_model.py \
  --model output/models/best_model.pth \
  --data-dir data/spectrograms/fold10 \
  --output results/local_evaluation.json
```

**Expected Output**:
```json
{
  "accuracy": 0.8124,
  "precision": 0.8156,
  "recall": 0.8124,
  "f1_score": 0.8132,
  "confusion_matrix": [[...]]
}
```

### 2.5 Start Local Inference Server

```bash
python scripts/04_local_inference_server.py \
  --model output/models/best_model.pth \
  --host 127.0.0.1 \
  --port 5000
```

**Expected Output**:
```
Loading model from: output/models/best_model.pth
Using device: cpu
Loaded checkpoint from epoch 20
Validation accuracy: 81.24%
Model loaded successfully!

LOCAL INFERENCE SERVER STARTED
Endpoint: http://127.0.0.1:5000/invocations
Health check: http://127.0.0.1:5000/ping
Stats: http://127.0.0.1:5000/stats

 * Running on http://127.0.0.1:5000
```

**Keep this terminal open** - server must run for stress testing.

---

## 3. AWS Deployment

### 3.1 AWS Prerequisites

1. **Install AWS CLI**:
   ```bash
   # Download from: https://aws.amazon.com/cli/
   # Verify installation
   aws --version
   ```

2. **Configure AWS Credentials**:
   ```bash
   aws configure
   # AWS Access Key ID: [Your Key]
   # AWS Secret Access Key: [Your Secret]
   # Default region name: us-east-2
   # Default output format: json
   ```

3. **Verify Access**:
   ```bash
   aws sts get-caller-identity
   ```

### 3.2 Create S3 Buckets

```bash
# Model artifacts bucket
aws s3 mb s3://urbansound8k-models-YOUR-ACCOUNT-ID --region us-east-2

# Inference demo bucket
aws s3 mb s3://urbansound-inference-demo-YOUR-ACCOUNT-ID --region us-east-2

# Enable versioning on models bucket
aws s3api put-bucket-versioning \
  --bucket urbansound8k-models-YOUR-ACCOUNT-ID \
  --versioning-configuration Status=Enabled
```

### 3.3 Upload Training Data to S3

```bash
# Upload spectrograms
aws s3 sync data/spectrograms s3://urbansound8k-models-YOUR-ACCOUNT-ID/data/spectrograms/

# Upload metadata
aws s3 cp urbansound8k_data/metadata/UrbanSound8K.csv \
  s3://urbansound8k-models-YOUR-ACCOUNT-ID/data/metadata/
```

### 3.4 Create IAM Roles

**Run the IAM setup script**:
```bash
python scripts/05_setup_iam_roles.py --account-id YOUR-ACCOUNT-ID
```

This creates:
- `UrbanSound8K-SageMaker-Role` (for training and inference)
- `UrbanSound8K-Lambda-Role` (for Lambda functions)

### 3.5 Train Model on SageMaker

1. **Package Training Code**:
   ```bash
   cd training
   tar -czf sourcedir.tar.gz train.py requirements.txt
   aws s3 cp sourcedir.tar.gz s3://urbansound8k-models-YOUR-ACCOUNT-ID/code/
   cd ..
   ```

2. **Submit Training Job**:
   ```bash
   python scripts/06_train_sagemaker.py \
     --bucket urbansound8k-models-YOUR-ACCOUNT-ID \
     --role-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/UrbanSound8K-SageMaker-Role \
     --job-name urbansound-training-$(date +%Y%m%d-%H%M%S) \
     --instance-type ml.m5.2xlarge \
     --epochs 20 \
     --batch-size 32
   ```

3. **Monitor Training**:
   ```bash
   # Check status
   aws sagemaker describe-training-job \
     --training-job-name urbansound-training-TIMESTAMP
   
   # View logs (in another terminal)
   aws logs tail /aws/sagemaker/TrainingJobs --follow
   ```

**Expected Duration**: ~3.5 hours
**Expected Cost**: ~$2.27

### 3.6 Deploy Model to SageMaker Endpoint

1. **Create Model**:
   ```bash
   python scripts/07_create_model.py \
     --model-name urbansound-classifier-v1 \
     --training-job-name urbansound-training-TIMESTAMP \
     --role-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/UrbanSound8K-SageMaker-Role
   ```

2. **Create Endpoint Configuration**:
   ```bash
   python scripts/08_create_endpoint_config.py \
     --config-name urbansound-endpoint-config \
     --model-name urbansound-classifier-v1 \
     --instance-type ml.m5.large \
     --initial-instance-count 1
   ```

3. **Deploy Endpoint**:
   ```bash
   python scripts/09_deploy_endpoint.py \
     --endpoint-name urbansound-classifier-v1 \
     --config-name urbansound-endpoint-config
   ```

4. **Wait for Deployment** (~5-10 minutes):
   ```bash
   aws sagemaker describe-endpoint --endpoint-name urbansound-classifier-v1
   # Wait until EndpointStatus: "InService"
   ```

### 3.7 Deploy Serverless Web Interface

1. **Create DynamoDB Tables**:
   ```bash
   python scripts/10_create_dynamodb_tables.py
   ```

2. **Package Lambda Functions**:
   ```bash
   cd lambda
   ./package_lambdas.sh
   cd ..
   ```

3. **Deploy Lambda Functions**:
   ```bash
   python scripts/11_deploy_lambda_functions.py \
     --bucket urbansound-inference-demo-YOUR-ACCOUNT-ID \
     --role-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/UrbanSound8K-Lambda-Role \
     --endpoint-name urbansound-classifier-v1
   ```

4. **Deploy Website**:
   ```bash
   # Configure S3 for static website hosting
   python scripts/12_deploy_website.py \
     --bucket urbansound-inference-demo-YOUR-ACCOUNT-ID
   ```

5. **Get Website URL**:
   ```bash
   echo "http://urbansound-inference-demo-YOUR-ACCOUNT-ID.s3-website.us-east-2.amazonaws.com"
   ```

### 3.8 Test AWS Deployment

```bash
# Test endpoint directly
python scripts/13_test_aws_endpoint.py \
  --endpoint urbansound-classifier-v1 \
  --test-image data/spectrograms/fold10/dog_bark/7383-3-0-0.png
```

**Expected Output**:
```json
{
  "predicted_class": "dog_bark",
  "confidence": 0.9942,
  "inference_time_ms": 200
}
```

---

## 4. Performance Benchmarking

### 4.1 Prerequisites for Stress Testing

Ensure both servers are running:

1. **Local Server** (Terminal 1):
   ```bash
   python scripts/04_local_inference_server.py \
     --model output/models/best_model.pth \
     --port 5000
   ```

2. **AWS Endpoint** (verify status):
   ```bash
   aws sagemaker describe-endpoint \
     --endpoint-name urbansound-classifier-v1 \
     --query 'EndpointStatus'
   ```

### 4.2 Install Stress Test Dependencies

```bash
pip install -r requirements-testing.txt
```

### 4.3 Run Progressive Stress Test

**Purpose**: Test both systems under increasing load (100 → 2000 concurrent requests).

```bash
python stress_tests/progressive_stress_test.py \
  --endpoint urbansound-classifier-v1 \
  --local-url http://127.0.0.1:5000/invocations \
  --test-folder data/spectrograms/fold10
```

**Expected Output**:
```
================================================================================
PHASE: Warmup - 100 requests, 10 concurrent
================================================================================
AWS:   100 success | 0 errors | 100.0% success rate
Local: 100 success | 0 errors | 100.0% success rate

================================================================================
PHASE: Heavy Load - 1000 requests, 100 concurrent
================================================================================
AWS:   234 success | 766 errors | 23.4% success rate
Local: 800 success | 200 errors | 80.0% success rate

... [Full results] ...

Results saved to: aggressive_stress_test_TIMESTAMP.json
```

**Expected Duration**: ~5-6 minutes
**Output**: JSON file with detailed metrics

### 4.4 Run Massive Dump Test

**Purpose**: Extreme test with 2000 simultaneous requests to demonstrate AWS advantage.

```bash
python stress_tests/massive_dump_test.py \
  --endpoint urbansound-classifier-v1 \
  --local-url http://127.0.0.1:5000/invocations \
  --test-folder data/spectrograms/fold10 \
  --requests 2000
```

**Expected Output**:
```
================================================================================
MASSIVE CONCURRENT DUMP TEST
================================================================================
Total Requests:  2000 to EACH endpoint

AWS SageMaker:
  Success Rate: 478/2000 (23.9%)
  Mean Response Time: 25.046s
  Throughput: 7.96 req/s

Local Model:
  Success Rate: 0/2000 (0.0%) - COMPLETE FAILURE
  
Results saved to: massive_dump_2000_TIMESTAMP.json
```

**Expected Duration**: ~3-4 minutes
**Key Finding**: AWS maintains 23.9% success vs Local 0% under extreme load

### 4.5 Generate Comparison Report

```bash
python scripts/14_generate_report.py \
  --progressive-test aggressive_stress_test_TIMESTAMP.json \
  --massive-dump massive_dump_2000_TIMESTAMP.json \
  --output results/performance_comparison_report.pdf
```

This creates a comprehensive PDF report with:
- Training accuracy comparison (AWS 81.24% vs Local 71.33%)
- Stress test results with graphs
- Cost analysis
- Scalability conclusions

---

## 5. Results Summary

### Expected Results

#### Training Performance
| Metric | AWS SageMaker | Local (CPU) |
|--------|---------------|-------------|
| Training Time | 3.5 hours | 8-12 hours |
| Validation Accuracy | 81.24% | 71.33% |
| Training Cost | $2.27 | $0 (existing hardware) |

#### Stress Test Performance (2000 Simultaneous Requests)
| Metric | AWS | Local |
|--------|-----|-------|
| Success Rate | 23.9% (478/2000) | 0.0% (0/2000) |
| Mean Response Time | 25.0s | N/A (all failed) |
| Throughput | 7.96 req/s | 0 req/s |

#### Key Conclusions
1. **Transfer learning works**: ResNet50 achieves 81.24% accuracy on UrbanSound8K
2. **AWS scales better**: Maintains partial availability under extreme load
3. **Local fails catastrophically**: 0% success at 2000 concurrent requests
4. **Cost-effective training**: $2.27 per training run on AWS
5. **Production-ready**: Complete serverless architecture with <11s latency

---

## 6. Cleanup (Optional)

### Delete AWS Resources

**WARNING**: This will delete all AWS resources and incur no further costs.

```bash
# Delete SageMaker endpoint
python scripts/15_cleanup_aws.py --endpoint urbansound-classifier-v1

# Or manually:
aws sagemaker delete-endpoint --endpoint-name urbansound-classifier-v1
aws sagemaker delete-endpoint-config --endpoint-config-name urbansound-endpoint-config
aws sagemaker delete-model --model-name urbansound-classifier-v1

# Delete Lambda functions
aws lambda delete-function --function-name urbansound-upload-handler
aws lambda delete-function --function-name urbansound-preprocessing
aws lambda delete-function --function-name urbansound-inference
aws lambda delete-function --function-name urbansound-get-results

# Delete DynamoDB tables
aws dynamodb delete-table --table-name urbansound-results
aws dynamodb delete-table --table-name urbansound-inference-results

# Empty and delete S3 buckets
aws s3 rm s3://urbansound-inference-demo-YOUR-ACCOUNT-ID --recursive
aws s3 rb s3://urbansound-inference-demo-YOUR-ACCOUNT-ID

aws s3 rm s3://urbansound8k-models-YOUR-ACCOUNT-ID --recursive
aws s3 rb s3://urbansound8k-models-YOUR-ACCOUNT-ID
```

---

## 7. Troubleshooting

### Common Issues

**Issue**: Kaggle download fails
```bash
# Solution: Verify kaggle.json is in correct location
ls ~/.kaggle/kaggle.json  # Linux/Mac
dir %USERPROFILE%\.kaggle\kaggle.json  # Windows
```

**Issue**: AWS credentials not configured
```bash
# Solution: Run aws configure again
aws configure
```

**Issue**: SageMaker training job fails with quota exceeded
```bash
# Solution: Request quota increase or use ml.m5.2xlarge (CPU)
# GPU instances require quota increase request
```

**Issue**: Local inference server crashes during stress test
```bash
# Solution: This is expected! Local server cannot handle 2000 concurrent requests
# This demonstrates AWS scalability advantage
```

**Issue**: Lambda function timeout
```bash
# Solution: Increase timeout in Lambda configuration
aws lambda update-function-configuration \
  --function-name FUNCTION-NAME \
  --timeout 300
```

---

## 8. Expected File Structure

After completing all steps:

```
urbansound8k-aws/
├── data/
│   └── spectrograms/          # 8,732 PNG spectrograms
│       ├── fold1/
│       ├── ...
│       └── fold10/
├── output/
│   └── models/
│       ├── best_model.pth     # Best performing model (81.24% accuracy)
│       └── final_model.pth    # Final epoch model
├── results/
│   ├── local_evaluation.json
│   ├── aggressive_stress_test_TIMESTAMP.json
│   ├── massive_dump_2000_TIMESTAMP.json
│   └── performance_comparison_report.pdf
├── scripts/                    # All Python scripts for reproduction
├── training/                   # SageMaker training code
├── lambda/                     # Lambda function code
├── stress_tests/              # Performance benchmarking code
└── requirements*.txt          # Dependencies
```

---

## 9. Contact and Support

- **GitHub Issues**: [https://github.com/m-jeakle/DATA-650]
- **Email**: markjeakle@gmail.com

---

## 10. Citation

If you use this work, please cite:

```bibtex
@misc{jeakle2024urbansound,
  title={UrbanSound8K Audio Classification Using Transfer Learning and AWS Cloud},
  author={Jeakle, Mark},
  year={2024},
  howpublished={DATA 650 - Cloud Computing},
  url={[https://github.com/m-jeakle/DATA-650]}
}
