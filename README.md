# Assignment 2

**Name:** Ragad Ziyada  
**Student ID:** 60301042  

---

## Project Overview
This assignment implements an end-to-end MLOps workflow for Amazon Electronics review sentiment classification using Azure Machine Learning and Azure DevOps. The pipeline automates model training, hyperparameter tuning, model registration, deployment, and inference evaluation.

---

## Model Choice
For this task, **Logistic Regression** was selected.

### Reasons for choosing Logistic Regression
- Efficient for high-dimensional sparse data (TF-IDF)
- Fast training suitable for CI/CD automation
- Interpretable linear decision boundary
- Works well for binary sentiment classification
- Lower compute cost compared to complex models

This makes it suitable for scalable Azure ML experiments and sweep jobs.

---

## Features Used
The model was trained using merged engineered features generated from the feature pipeline:

### 1. SBERT Embeddings
- Dense semantic representation of review text
- Captures contextual meaning
- Improves semantic understanding

### 2. TF-IDF Features
- Sparse bag-of-words representation
- Captures word frequency importance
- Works well with linear models

### 3. Sentiment Features
- Numeric sentiment polarity score
- Helps capture emotional tone

### 4. Review Length Features
- Review length statistics
- Useful for distinguishing informative reviews

### Final Feature Representation
All features were concatenated into a single feature matrix before model training.

---

## Hyperparameter Tuning
Hyperparameter tuning was performed using Azure ML Sweep Job.

### Tuned Parameters
- Regularization strength (C / alpha)
- Maximum iterations

### Sweep Configuration
- Sampling algorithm: Random
- Objective metric: Validation accuracy
- Trials: Multiple runs on CPU cluster

### Best Hyperparameters (from sweep):

- --C = 5.724181499928872  
- max_iter = 300  

These parameters were then used in the final training job.

---

## Model Performance

### Training Metrics
- Training Accuracy: (from MLflow)
- Validation Accuracy: (from MLflow)
- Test Accuracy: (from MLflow)


### Additional Metrics Logged
- Precision
- Recall
- F1 Score
- Training runtime (seconds)

All metrics were logged using MLflow and tracked in Azure ML experiments.

---

## Experiment Comparison
Three feature configurations were tested:

- SBERT only
- SBERT + TF-IDF
- All features

Best performance was achieved using **all features combined**, demonstrating complementary feature contributions.

---

## Training Pipeline
The training pipeline:

- Loads merged feature datasets
- Creates binary labels
- Builds feature matrix
- Trains Logistic Regression model
- Logs metrics using MLflow
- Saves model artifact (model.pkl)

---

## Azure DevOps Automation
Azure DevOps pipeline automatically:

- Connects to Azure ML workspace
- Submits training job
- Streams logs
- Tracks experiment

Pipeline triggered on push to: assignment-2


---

## Model Registration
The final trained model was registered in Azure ML Model Registry to enable:

- version tracking
- reproducibility
- deployment

---

## Deployment
The registered model was deployed to:

- Azure ML Managed Online Endpoint
- REST API for real-time predictions
- Deployment status: Healthy

---

## Endpoint Evaluation
The deployed model was invoked using the deployment dataset.

Metrics observed:

- Successful predictions
- Low latency
- Deployment accuracy close to test accuracy

This confirms the model generalizes to new unseen data.


<img width="1896" height="870" alt="Screenshot 2026-03-29 094802" src="https://github.com/user-attachments/assets/e6512fac-3b3e-4a9a-932b-3a082d1107e6" />
<img width="1811" height="725" alt="Screenshot 2026-03-30 213057" src="https://github.com/user-attachments/assets/49be6d32-3a89-42a5-9878-3d9802ecb1fd" />
<img width="761" height="203" alt="Screenshot 2026-03-30 235833" src="https://github.com/user-attachments/assets/5d282f49-bbd8-4ba7-a641-f5340f6b7c14" />
<img width="1893" height="825" alt="Screenshot 2026-04-01 121613" src="https://github.com/user-attachments/assets/29c08dc6-e4f2-49dc-a9be-5343b7017b5e" />
<img width="1857" height="875" alt="image" src="https://github.com/user-attachments/assets/74c60d0c-c475-4028-b538-12e12c857b75" />



