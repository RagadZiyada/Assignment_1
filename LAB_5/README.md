# LAB 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

<img width="1908" height="881" alt="Screenshot 2026-03-10 185830" src="https://github.com/user-attachments/assets/a4d4cbe9-78fa-453f-8585-192ff1fbdef8" />




This lab implements a complete machine learning pipeline for predictive maintenance using the NASA C-MAPSS FD001 dataset. The goal of the project is to predict the Remaining Useful Life (RUL) of aircraft engines based on multivariate sensor time-series data.



The pipeline was designed to run both locally and on Azure Machine Learning, demonstrating how machine learning workflows can be scaled and executed reproducibly in a cloud environment.





---



# Pipeline Workflow



The pipeline is composed of several sequential stages that transform raw time-series data into a trained predictive model.



---



## 1. Data Preparation



The NASA FD001 dataset contains time-series sensor readings for multiple engines over their operational cycles.



During this step:



- raw training and test data are loaded

- column names are assigned

- Remaining Useful Life (RUL) is calculated for each engine cycle

- cleaned datasets are saved for further processing



This creates the target variable needed for model training.



---



## 2. Feature Extraction



The \*\*tsfresh\*\* library is used to automatically extract statistical features from the time-series sensor signals.



Instead of using raw sensor readings, tsfresh generates descriptive features such as:



- mean

- variance

- autocorrelation

- frequency-based statistics



These features capture patterns in the engine behavior over time.



---



## 3. Feature Filtering



Because feature extraction can generate a very large number of features, several filtering techniques are applied:



- \*\*Variance Threshold\*\* removes features with very low variability

- \*\*Correlation filtering\*\* removes highly redundant features

- \*\*Mutual Information ranking\*\* keeps only the most informative features



This step reduces dimensionality and improves model efficiency.



---



## 4. Genetic Algorithm Feature Selection



A \*\*Genetic Algorithm (GA)\*\* is used to further refine the feature set.



The GA simulates evolutionary processes to search for feature combinations that produce the best prediction performance.



Each candidate solution is evaluated using cross-validated RMSE, and the algorithm evolves toward better feature subsets.



---



## 5. Model Training



A \*\*LightGBM regression model\*\* is trained using the selected features.



LightGBM was chosen because it is efficient for large datasets and performs well on structured data.



The model predicts the Remaining Useful Life of engines and is evaluated using:



- RMSE (Root Mean Squared Error)

- MAE (Mean Absolute Error)

- R² Score



---



# Azure Machine Learning Execution



The pipeline was executed as an \*\*Azure Machine Learning command job\*\*.



Azure ML provides:



- reproducible environments

- scalable compute resources

- experiment tracking

- centralized logging



Running the pipeline in Azure ensures the workflow can scale to larger datasets and be executed consistently.



The job logs show:



- feature extraction progress

- selected feature counts

- model training results

- execution runtime



---



# Running the Pipeline Locally



The pipeline can be executed locally using:





python src/run\_pipeline.py --data\_dir data --output\_dir outputs\_local





---



# Running the Pipeline on Azure



To run the pipeline on Azure Machine Learning:



az ml environment create --file environments/lab5\_env.yml

az ml data create --file data\_assets/fd001.yml

az ml job create --file jobs/lab5\_command\_job.yml







Azure will execute the full pipeline and store logs and outputs.



---



# Key Results



The pipeline successfully:



- extracted statistical time-series features

- selected the most informative features

- trained a predictive maintenance model

- evaluated prediction performance using RMSE



Azure job logs also provide execution time and pipeline progress.



---



# Conclusion



This lab demonstrates how automated feature extraction, feature selection techniques, and cloud-based machine learning infrastructure can be combined to build an effective predictive maintenance pipeline.



Using Azure Machine Learning makes the workflow reproducible, scalable, and suitable for real-world industrial applications involving large sensor datasets.

