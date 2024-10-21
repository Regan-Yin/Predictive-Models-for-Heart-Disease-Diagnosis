# Predictive Models for Heart Disease Diagnosis

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Data Preparation](#2-data-preparation)  
    2.1 [Data Understanding](#21-data-understanding)  
        2.1.1 [Dataset Description](#211-dataset-description)  
        2.1.2 [Label Determination](#212-label-determination)  
        2.1.3 [Feature Selection](#213-feature-selection)  
        2.1.4 [Feature Description](#214-feature-description)  
        2.1.5 [Data Visualization](#215-data-visualization)  
    2.2 [Data Preprocessing](#22-data-preprocessing)  
        2.2.1 [Categorical and Numerical Variables](#221-categorical-and-numerical-variables)  
        2.2.2 [Variables Conversion](#222-variables-conversion)  
    2.3 [Dataset Split](#23-dataset-split)  
3. [Model Building](#3-model-building)  
    3.1 [Select the right machine learning models](#31-select-the-right-machine-learning-models)  
    3.2 [Build models](#32-build-models)  
4. [Performance Evaluation](#4-performance-evaluation)  
5. [Conclusion](#5-conclusion)  

## 1. Introduction

Heart disease is one of the leading causes of death worldwide. Its symptoms differ from person to person. It is possible to have mild symptoms or even no symptoms at all, which means that many individuals do not know they have the disease until a critical event happens. Therefore, early prediction and detection of heart disease, followed by preventative measures, are essential.

Due to multiple risk factors, traditional programming and heuristics may not be sufficient to identify individuals at high risk. This is where machine learning techniques come into play. By analyzing large amounts of data, machine learning models can identify patterns and correlations that may not be apparent to humans. These models can predict the likelihood of an individual developing heart disease, enabling early intervention. In this project, we apply data mining techniques and build predictive models for heart disease prediction using the 2020 BRFSS dataset.

## 2. Data Preparation

### 2.1 Data Understanding

#### 2.1.1 Dataset Description

The dataset we use is the 2020 BRFSS data (ASCII) from the Behavioral Risk Factor Surveillance System (BRFSS), which monitors modifiable risk behaviors and other factors contributing to mortality in the population. The dataset includes 279 attributes and responses from 401,958 individuals, covering health status, exercise, chronic health conditions, and more. We utilize this dataset to predict heart disease.

#### 2.1.2 Label Determination

We choose the variable “_MICHD” as our target label, which indicates if an individual has ever reported coronary heart disease (CHD) or myocardial infarction (MI). We treat those who reported having CHD or MI as positive cases (label 1) and those who did not as negative (label 0).

#### 2.1.3 Feature Selection

We start by handling missing values. Features with more than 90% missing values are dropped. We also remove irrelevant or repeated features, leaving us with 97 variables—17 numerical and 80 categorical variables.

#### 2.1.4 Feature Description

The selected variables include numerical variables and categorical variables which are summarized and processed as required for the model.

#### 2.1.5 Data Visualization

The dataset contains 34,163 individuals diagnosed with heart disease, making up 8.5% of the total. Gender-wise, the dataset includes 45.76% male and 54.24% female respondents.

### 2.2 Data Preprocessing

#### 2.2.1 Categorical and Numerical Variables

We split the data into categorical and numerical variables, handling missing or special codes (e.g., 7, 9, 77, 98) by converting them to mode or mean values. Features with blank values are handled accordingly based on the nature of the blank.

#### 2.2.2 Variables Conversion

We apply One-Hot Encoding to categorical variables, converting them into dummy variables (binary values).

### 2.3 Dataset Split

We split the dataset into a training set and a testing set, using 40% of the data for testing. This split ensures consistency in results, with the training set used for model training and the testing set for evaluation.

## 3. Model Building

### 3.1 Select the right machine learning models

Our task is to predict heart disease (a binary classification). We choose supervised learning algorithms: decision tree, logistic regression, k-nearest neighbor, and naive bayes.

### 3.2 Build models

#### 3.2.1 Step 0: Determine the cost for false-positive and false-negative

We prioritize minimizing false negatives (failing to detect heart disease) over false positives (incorrectly predicting heart disease). Thus, we set the cost of false negatives to 100 and false positives to 10.

#### 3.2.2 Step 1: Determine hyperparameters for tuning

We tune model hyperparameters like max_depth, max_leaf_nodes, regularization strength (C), and others. Default settings are used where appropriate.

#### 3.2.3 Step 2: Apply GridSearchCV for hyperparameter tuning

We use GridSearchCV with 10-fold cross-validation to find the optimal set of hyperparameters. We focus on recall to minimize false negatives.

#### 3.2.4 Step 3: Train the model

We train each model with the selected hyperparameters on the training set, then evaluate them on the testing set using metrics like accuracy and confusion matrix.

#### 3.2.5 Step 4: Find the optimal decision threshold

We minimize the total cost using the cost function:  
`Total cost = 100 * False Negatives + 10 * False Positives`

#### 3.2.6 Step 5: Generate new outcomes

With the optimal decision threshold, we adjust predictions and re-evaluate model performance.

## 4. Performance Evaluation

Before adjusting the decision threshold, logistic regression had the highest accuracy and lowest MAE, while naive bayes had the highest recall. After adjusting, logistic regression still performed the best in recall, MAE, and cost, making it the most suitable model for prediction.

## 5. Conclusion

In this project, we built and evaluated four predictive models for heart disease diagnosis. The logistic regression model outperformed others in terms of accuracy and recall, making it the most effective for identifying patients at risk. However, future work should consider refining the cost functions to better reflect real-world implications.
