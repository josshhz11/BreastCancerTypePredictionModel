
# Breast Cancer Type Prediction Model

![image](https://github.com/user-attachments/assets/a94ee5df-20a6-441e-b8cd-648e133b3cb4)

#### SC1015 Mini Project
NTU School of Computer Science and Engineering
- Lab Group - **ACDA1, Group 6**

Members
- [Joshua Foo Tse Ern](https://github.com/josshhz11)
- [Kieran Voo E Kai](https://github.com/kieranvoo)
- Yuan Haoer

Using Jupyter Notebook and Python 3.7

## Problem Statement
Predicting the types of breast cancer based on the provided genes in the dataset.

## Dataset used
[Breast cancer gene expression - CuMiDa](https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida)

### Topic covered in this notebook
1. Data Preparation and Cleaning
2. Exploratory Data Analysis
3. ML Models Analysis
    - K-Nearest Neighbours Classification
    - Random Forest Classification
    - Support Vector Model (SVM) Classification
4. Model Evaluation
5. Data Driven Insights and Recommendation

## Introduction and Problem Statement
Breast cancer is the most common cancer worldwide and continues to rise. In Singapore alone, it has caused 270 deaths annually and has seen a 160% increase in incidence over the past four decades, underscoring the importance of this subject topic for society.

Breast cancer involves multiple genetic mutations in proto-oncogenes and tumor suppressor genes, which disrupt normal cell cycle control, causing uncontrollable cell division and forming a tumor.

Thus, our group seeks to explore the genes that determine and affect the type of breast cancer present in women. In doing so, we hope that the results of the model we have created will help women find out the likelihood of the certain types of breast cancer they could potentially have as per their genes, and thereby aid them in determining what type of prevention or treatment is required accordingly.

## Data Cleaning

#### Introduction to our dataset
Our original dataset from Kaggle contained a total of 151 rows and 54,677 columns, where we have 151 patients and their respective types of breast cancer (or lack thereof), accompanied by the genes they had.

#### Removing of certain breast cancer types 
In beginning our data cleaning, we first removed all rows with the values 'cell_line' and 'normal' under the column 'types'. This is because 'cell_line' refers to breast cancer that was developed in lab experiments **for testing**, and 'normal' refers to tests where breast cancer was **not present**. As such, we removed 'cell_line' as breast cancer that was from lab experiments are not an accurate representation of the breast cancer types we are trying to model, and we removed 'normal' since tests without breast cancer will not aid our exploration into the types of breast cancer caused by different types of genetics.

#### Checking of duplicates and missing values
We then checked for any duplicate and missing values throughout the entire dataset, to which we did not find any at all, requiring no cleaning on that front.

#### Cleaning of gene dataset
However, the number of genes in the dataset was too many for such an exploration, with over 54,600. As such, we decided that we will be using **PCA** as a method to select the genes that are most likely to help us in our exploration in predicting the various types of breast cancer. These genes from the selection will serve as our predictors for the response variable of types of breast cancer in our exploration. 

#### Elaborate on PCA and how we removed some other genes from the PCA gene dataset.

PCA is a statistical technique to emphasize variation and identify strong patterns in a dataset. Due to the use of high dimensional data, we utilised PCA to convert our possible correlated variables into a set of values of linearly uncorrelated values called principal components. Principal components are ordered by the amount of variance they capture from the dataset. PCA loadings represent how much each gene contributes to each principal component. By taking the gene with the maximum loading in each of the top 15 principal components, we effectively extracted the genes that will contribute most to the dataset's variability. We will be using these genes in our predictive models.

#### Further cleaning of the PCA gene predictors
After we obtained our 15 genes for use in our exploration through the PCA, we imported the *mygene* library to compare our 15 gene predictors. This was to find out if they were actual genes that exist and genes that can be mapped to gene symbols. We also further researched if the genes that we found among the 15 PCA gene predictors were potentially related to cancer in any way.

From this above step, we found that 3 out of the 15 genes could not be found within the *mygene* library, and we decided to remove these 3 genes as predictors from our set of PCA gene predictors, leaving us with just 12 gene predictors remaining. For those we could identify, we were able to find a reported correlation between the respective genes and cancer.

We believe that due to lack of potential understanding of the 3 genes due to their unknown natures, they may inaccurately affect our exploration due to their potential lack of correlation with breast cancer at all.

However, to ensure that the genes picked out by PCA are good predictors of the types of breast cancer, we will conduct an evaluation of these PCA genes against an equal number of randomly selected genes from the dataset.

## Exploratory Data Analysis
To get a better understanding of our data, we plotted the box plots of the 12 genes we have selected from the PCA method, to understand the gene spread across the various subjects, and number of outliers. Although there are outliers, we will not remove them due to the thin nature of our dataset (lack of data entries) and also due to these outliers possibly containing important biological information which are important in helping us predict the cancer type.

Box Plots:
![image](https://github.com/user-attachments/assets/8acb9438-0d3e-4883-9fe3-3ef33c9c32cc)

We plotted a correlation matrix to find the correlations between gene pairs and any potential relationship between them, to which some strong positive and negative correlations between gene pairs surfaced.

![image](https://github.com/user-attachments/assets/b32faea6-9628-435c-8e69-c4d6c545a8b8)

We also plotted pair plots and violin plots to obtain a more comprehensive stance on the clustering of genes as well as the varied densities across different genes.

Pair Plots:
![image](https://github.com/user-attachments/assets/4f982e35-a7df-4025-9231-a735c41c3b0d)

Violin Plots:
![image](https://github.com/user-attachments/assets/b9b4a9c5-eeda-49ee-9610-c40ffb003b36)

From our EDA, we concluded that the minimal amount of outliers and the correlations between the chosen gene predictors meant that these gene predictors selected were suitable to use in our model.

## ML Models Analysis
To evaluate our classification models, we will be using accuracy, precision, recall and f1 score. We cannot solely depend on the accuracy metric since our data might be underfit or overfit. We will be using confusion matrix to see the amount of **True Positive** and **True Negative** which our model correctly predict the outcomes.

- **Accuracy**: Proportion of true results among the total number of cases examined
- **Precision**: Accuracy of positive predictions
- **Recall**: Fraction of positives that were correctly identified
- **F1 score**: What percent of positive predictions were correct?

Below, we will be using classification models such as K-Nearest Neighbours (KNN), Random Forest (RF) Classification, and Support Vector Model (SVM) Classification.


### 1. K-Nearest Neighbours (KNN) Classifier
The K-Nearest Neighbours (KNN) classifier is a type of instance-based learning algorithm used in machine learning that classifies new cases based on a similarity measure (usually distance functions) with known cases from the training dataset, typically voting based on the 'k' most similar instances.

Goodness of Fit of KNN Model (PCA) - Test Dataset:
- Accuracy: 0.62
- Precision: 0.76
- Recall: 0.59
- F1 Score: 0.62

Goodness of Fit of KNN Model (Random) - Test Dataset:
- Accuracy: 0.54
- Precision: 0.62
- Recall: 0.53
- F1 Score: 0.52

### 2. Random Forest (RF) Classifier
A random forest (RF) classifier is a type of ensemble learning algorithm used for classification tasks in machine learning. It is called a "forest" because it consists of multiple decision trees, where each tree is trained on a different subset of the input data, and the final prediction is made by aggregating the predictions of all the trees.

Goodness of Fit of RF Model (PCA) - Test Dataset:
- Accuracy: 0.73
- Precision: 0.79
- Recall: 0.73
- F1 Score: 0.70

Goodness of Fit of RF Model (Random) - Test Dataset:
- Accuracy: 0.50
- Precision: 0.49
- Recall: 0.47
- F1 Score: 0.48

### 3. Support Vector Model (SVM) Classifier
The Support Vector Machine (SVM) classifier is a powerful, supervised machine learning algorithm used for both classification and regression tasks. It works by finding the hyperplane that best separates different classes in the feature space, with support vectors being the critical elements of the training dataset that influence the position of the hyperplane.

Goodness of Fit of SVM Model (PCA) - Test Dataset:
- Accuracy: 0.77
- Precision: 0.82
- Recall: 0.82
- F1 Score: 0.76

Goodness of Fit of SVM Model (Random) - Test Dataset:
- Accuracy: 0.50
- Precision: 0.47
- Recall: 0.47
- F1 Score: 0.46

## Model Evaluation
Based on the models above, we can deduce that **Support Vector Model (SVM) Classifier** is the best model to predict the the types of breast cancer based on PCA-selected genes. Accuracy, Precision, Recall and F1 score are all uplifted in both PCA and randomly-selected genes, showing the power of the model and not just the power of the PCA-selected genes. For the PCA-selected genes, the accuracy, precision i, recall and F1 scores hover around the 76% to 82% range.

## Data-Driven Insights and Recommendation

In terms of data-driven insights, for all three models, the most misclassifications were made between luminal_A and luminal_B. **More samples would be needed for more accurate classification**, to distinguish between the two types. We thus suggest that in future work, more observations can be included to improve the accuracy of any model selected. Driven by our curiosity to further investigate this interesting phenomenon, we conducted a bit more research and found that luminal A and B cancer types are very similar in terms of causes, manifestations, and treatments. This is in contrast with HER and basal cancer types, which have their own defining characteristics.

Overall, we found that the top genes generated by Principal Component Analysis (PCA) do indeed have a **stronger predictive effect on the cancer type** as compared to the rest of the genes, represented by a random sample. We also found that Support Vector Machine (SVM) is the best model for our analysis, as it has the highest accuracy and least classification errors. Combining these two insights, we can now conclude that it is possible to reliably predict the cancer type by using SVM model to analyse data of genes obtained from PCA.

## Acknowledgements
 - [Scikit Learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
 - [Pandas](https://pandas.pydata.org/)
 - [Seaborn](https://seaborn.pydata.org/)
 - [WCRFI](https://www.wcrf.org/cancer-trends/breast-cancer-statistics/)
 - [NLM](https://pubmed.ncbi.nlm.nih.gov/19701678/)
 - [Kaggle](https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida/)
 - [NLM](https://www.ncbi.nlm.nih.gov/books/NBK583808/)

 ## License
[MIT](https://choosealicense.com/licenses/mit/)


## Badges
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

