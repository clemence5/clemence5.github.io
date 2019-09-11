---
title: "Absenteeism in the Workplace: Preprocessing"
date: 2019-09-10
tags: [Preprocessing]
excerpt: "Machine Learning, Absenteeism, Business, Data Science"
header:
  overlay_image: "/images/absenteeism/out-office-760.jpg"
  overlay_filter: 0.5
  caption: "Credit: [**CallCentreHelper**](https://www.callcentrehelper.com/managing-absenteeism-in-the-call-centre-168.htm)"
  actions:
    - label: "View on Google Colab"
      url: "https://colab.research.google.com/drive/1VU3g2zU0iHDL5TFgbjxKee5Fc_cW3AEi"
mathjax: true
---

## Preparing Absenteeism data for inspection via ML.
In this exercise, we go through the multiple columns in the dataset and prepare them for the Machine Learning algorithms we will later deploy to find out what intuitions are buried in the data.

Later, we will develop a ML algorithm, the `Logistic Regression` and analyse the results with the aim of predicting when and why typically miss work in a business environment.
This helps businesses save a lot of money by anticipating absenteeism and deploying measures that minimise the resulting loss in productivity.  

The preparation involves extracting relevant data, dropping irrelevant columns and converting the data types in columns to usable format.

### The Dataset
The data, and some of the code used in this article are adapted from [Data Science Course](https://www.udemy.com/course-dashboard-redirect/?course_id=1754098) on Udemy.

# The dataset will be uploaded on GitHub on the link below
### Loading the dataset:
We begin by importing the relevant libraries: `numpy`, which allows us to work with numbers and manipulate arrays with ease, and `pandas`, which is designed for working with `*pa*nel *da*ta`.
```python
import numpy as np  # For array manipulation
import pandas as pd # For easily viewing and manipulating dataframes
dataset = '/content/gdrive/My Drive/Colab Notebooks/Absenteeism-data.csv'
raw = pd.read_csv(dataset, sep=',')
raw.head()
```
## Data inspection
<img src="{{ site.url }}{{ site.baseurl }}/images/absenteeism/absenteeism_columns.png" width="2500" height="2300" alt="Raw Data header">

From the file header we see the columns from left to right and data they contain in each row.

A `Logistic Regression` is an equation of the 

We can also get an initial idea of the dataset by displaying a concise summary of the data contained in each column:
```python
raw.info()
```  
<img src="{{ site.url }}{{ site.baseurl }}/images/absenteeism/raw_data_info.png" width="2500" height="2300" alt="Raw Data header">
This shows us that all the columns have 700 rows of data, as well as the data types, which are mostly integers, save for one float column and the `Date` column, which contains string entries, which in `Pandas` are referred to as `Objects`.
### Dropping the `ID` column
From the outset, we can see that the `ID` column will not be useful for the ML algorithm deployment, since we are not necessarily interested in the individual employees, but their behaviour in general.

We thus drop this column. But first, let's save a copy of our raw, unaltered dataframe that we can always revert to in the case of a mistake.

```python
df = data.copy()
df = df.drop(['ID'], axis=1)
print(np.shape(df))
```
Now, we see that we are dealing with 11 columns containing information relevant to our task. Each of the columns contains 700 observations.

## Reason for Absence
This column contains information on the employees' self-reported reasons for non-attendance.
