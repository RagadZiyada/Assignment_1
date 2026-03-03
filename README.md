# Assignment\_1

\# Lab 4 – Text Feature Engineering with Azure ML
<img width="1897" height="869" alt="image" src="https://github.com/user-attachments/assets/a1eaf8ff-0a30-4de8-bd9e-2e5da7590c66" />
<img width="441" height="618" alt="image" src="https://github.com/user-attachments/assets/f6d60fef-8d0b-4d7f-ab30-03ecd0096342" />

This lab focuses on transforming Amazon Electronics review text into machine learning features using Azure Machine Learning components and pipelines. The work builds on the curated Gold dataset created in the previous lab and uses a sampled version of the dataset to make feature engineering more computationally manageable.



## Objective



The goal of this lab was to inspect the curated review dataset, validate the schema and data quality, create a sampled subset, engineer text-based features using Azure ML components, and register the final output in the Azure ML Feature Store.



## Dataset Exploration and Validation



The first part of the lab was completed in Databricks. The curated Gold dataset (`features\_v1`) was loaded and inspected to confirm that:



- review text fields such as `reviewText` were stored as strings

- rating values such as `overall` were numeric

- entity identifiers such as `asin` and `reviewerID` were present

- no major malformed data types appeared in the key columns



Missing and empty values were also checked for important columns like `reviewText`, `overall`, `asin`, and `reviewerID`.


## Azure ML Feature Engineering Pipeline



The second part of the lab was implemented using Azure ML command components. Each component was designed to perform one specific task only.



### Components Created



#### 1. Split Dataset

This component split the sampled dataset into:

- training set

- validation set

- test set



This was important to avoid data leakage, especially for TF-IDF fitting.



#### 2. Normalize Review Text

This component cleaned the review text by:

- converting text to lowercase

- removing punctuation

- replacing URLs

- replacing numbers

- trimming whitespace

- filtering out very short reviews



This step ensured that later features were built on cleaner and more consistent text.



#### 3. Review Length Features

This component created:

- `review\_length\_words`

- `review\_length\_chars`



These basic features capture how long a review is, which can sometimes correlate with review quality or information richness.



#### 4. Sentiment Features

This component used sentiment analysis to create:

- `sentiment\_pos`

- `sentiment\_neg`

- `sentiment\_neu`

- `sentiment\_compound`



These features capture emotional tone and polarity in the reviews.



#### 5. TF-IDF Features

This component used TF-IDF vectorization to create sparse text features based on word importance. The vectorizer was fit only on the training split and then applied to validation and test splits to avoid leakage.



#### 6. SBERT Embeddings

This component created dense semantic embeddings from the review text using Sentence-BERT. These features capture deeper contextual meaning beyond simple word frequency.



#### 7. Merge Features

This component merged all engineered features into one final feature-enriched dataset using the entity keys:

- `asin`

- `reviewerID`



## Environment and Reproducibility



Each Azure ML component was defined using:

- a Python script

- a `component.yml` file

- a `conda.yml` file



Using explicit environments helped ensure the required packages such as `pandas`, `pyarrow`, `scikit-learn`, `nltk`, and `sentence-transformers` were available and consistent across runs.



## Pipeline Execution



All components were connected into a single Azure ML pipeline. The pipeline performed the following sequence:



1\. load sampled dataset

2\. split into train/validation/test

3\. normalize text

4\. generate review length features

5\. generate sentiment features

6\. generate TF-IDF features

7\. generate SBERT embeddings

8\. merge all features into one dataset



The pipeline completed successfully in Azure ML.



## Feature Store Registration



After the pipeline completed, the merged output from the `merge\_all` step was used to define and register a versioned feature set in the Azure ML Feature Store.



The entity used was:



- `AmazonReview`



The final feature set registered was:



- `amazon\_review\_text\_features`



## Challenges Faced



This lab initially had a few setup-related issues, including:

- missing component files in the new repository

- some components not yet having explicit environments

- Azure ML input handling requiring folder-aware parquet loading

- a small CLI command issue while registering the feature store entity



These issues were fixed by creating the full component structure, adding `conda.yml` files, and updating the component scripts to correctly resolve parquet input files.



## Final Outcome



At the end of this lab, I successfully:

- explored and validated the curated dataset

- created a sampled dataset

- built Azure ML feature engineering components

- registered all components

- ran the full Azure ML pipeline successfully

- produced merged feature output

- prepared the assets needed for Feature Store registration



This lab provided a strong introduction to reproducible text feature engineering workflows using Azure ML pipelines and Feature Store assets.

