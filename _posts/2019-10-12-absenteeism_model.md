---
title: "Absenteeism in the Workplace: Building a model to predict absenteeism"
date: 2019-10-12
tags: [Logistic regression, Logit, Classification]
excerpt: "Machine Learning, Absenteeism, Business, Data Science"
header:
  overlay_image: "/images/absenteeism/out-office-760.jpg"
  overlay_filter: 0.5
  caption: "Credit: [smallbusiness.co.uk](https://s17026.pcdn.co/wp-content/uploads/sites/9/2017/06/Out-of-office-8617.jpeg)"
  actions:
    - label: "View on Google Colab"
      url: "https://colab.research.google.com/drive/1RHBq5TXhqnvfiTRf1ScozQD0wLAN0xf2#scrollTo=kuW_zRfeQ-t5"
mathjax: true
---

### Summary of current progress:

In the [previous article](https://clemence5.github.io/absenteeism/) we **preprocessed** our absenteeism dataset by performing the following steps:

* Dropping the `ID` column (it contained no useful information for our upcoming analysis)
* Performed some exploratory analysis on the `Reason for absence` column, which contained integers describing the reasons for absenteeism. We performed `dummy encoding` and grouped the dummy variables into 4 classes:
 * various diseases
 * pregnancy-related reasons
 * poisoning
 * light reasons
 and replaced the original column with four dummy-encoded reasons columns
* Split the date column into month and weekday columns
* Grouped the education column into two classes representing High School and Tertiary-level graduates respectively.
* Finally, we saved our preprocessed dataset as `absenteeism_data_preprocessed.csv` in our working directory.

### So, What's next?

Now, we would like to use our preprocessed data to build a `logistic regression classifier`, or `Logit model`, to help us predict whether or not a given employee will exhibit *excessive absenteeism*, based on information encoded in the predictors we preprocessed.

Let us begin by importing the usual libraries, loading and taking a preliminary look at our preprocessed data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set('notebook')
from IPython.display import display, Image, SVG, Math

%matplotlib inline

dataset ='absenteeism_data_preprocessed.csv'
raw = pd.read_csv(dataset)
raw.sample(5)
```
