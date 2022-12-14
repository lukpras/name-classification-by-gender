# -*- coding: utf-8 -*-
"""gender-classification-by-name.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IqpA28upMOcEZWUqYO_C_uekoc8qLdFv

# **Gender Name Classification**
## **Luki Prasetyo**
### *https://www.dicoding.com/users/lukiprasetyo*
1nd Submission Task for "Machine Learning Developer - Machine Learning Terapan"

## Importing Dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
warnings.filterwarnings("ignore")
# %matplotlib inline

"""## Load Dataset"""

!wget --no-check-certificate \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00591/name_gender_dataset.csv

df = pd.read_csv(r'name_gender_dataset.csv')
df.sort_values(by='Name', ascending=True)

df.describe()

df.info()

"""## Data Visualization"""

sns.countplot('Gender', data=df)

df['Gender'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', textprops={'color':"w", 'fontsize':'20'},shadow=True)

"""```
There is a significant difference between Men and Women regardless of the number in each name available
```
"""

df2 = df.groupby('Gender').sum().reset_index()
df2.head()

sns.barplot(x="Gender", y="Count", data=df2)

df2['Count'].plot.pie(autopct='%1.2f%%', textprops={'color':"w", 'fontsize':'20'},shadow=True, labels = ["Female", "Male"])

"""```
If we look at the comparison between men and women with the number in each name, then the difference in the number is the same between men and women.
```

## Data Preparation and Feature Engineering

### Categorize Data
```
add a column to categorize different names into a male or female bucket based on whether or not the frequency of males for a name outnumbers the frequency of females.
```
"""

namechart = df.reset_index().pivot('Name', 'Gender', 'Count')
namechart = namechart.fillna(0)
namechart["percent_male"] = ((namechart["M"] - namechart["F"])/ (namechart["M"] + namechart["F"]))
namechart['gender'] = np.where(namechart['percent_male'] > 0.001, 'Male', 'Female')
namechart

"""### Transforming Text to Vector"""

char_v = CountVectorizer(analyzer='char', ngram_range=(2, 2))
X = char_v.fit_transform(namechart.index)
X = X.tocsc()
y = (namechart.gender == 'Male').values.astype(np.int)
print(X)

"""### Splitting train and validation sets"""

train, test = train_test_split(range(namechart.shape[0]), train_size=0.7, random_state=25)
mask=np.ones(namechart.shape[0], dtype='int')
mask[train]=1
mask[test]=0
mask = (mask==1)

"""## Creating and Training the Model"""

X_train=X[mask]
y_train=y[mask]
X_test=X[~mask]
y_test=y[~mask]

MNB = MultinomialNB(alpha=1)
MNB.fit(X_train, y_train)
NB_train_acc = round(MNB.score(X_train, y_train) *100, 2)
NB_test_acc = round(MNB.score(X_test, y_test) *100, 2)
        
print(NB_train_acc)
print(NB_test_acc)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
log_train_acc = round(logreg.score(X_train, y_train) * 100, 2)
log_test_acc = round(logreg.score(X_test, y_test) * 100, 2)

print(log_train_acc)
print(log_test_acc)

DT = DecisionTreeClassifier(max_depth=15, random_state=42)
DT.fit(X_train, y_train)
DT_train_acc = round(DT.score(X_train, y_train) * 100, 2)
DT_test_acc = round(DT.score(X_test, y_test) * 100, 2)

print(DT_train_acc)
print(DT_test_acc)

models = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression', 'Decision Tree'],
    'Train_Score': [NB_train_acc, log_train_acc, DT_train_acc],
    'Test_Score': [NB_test_acc, log_test_acc, DT_test_acc]})
models.sort_values(by='Train_Score', ascending=False)

param_grid = {
    'penalty' : ['l1', 'l2'],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
}

LR = LogisticRegression()
clf = GridSearchCV(LR, param_grid = param_grid, cv = 25, verbose=3, scoring='accuracy')

best_clf = clf.fit(X_train, y_train)

print('Best Penalty:', best_clf.best_estimator_.get_params()['penalty'])
print('Best C:', best_clf.best_estimator_.get_params()['C'])
print('Best Solver:', best_clf.best_estimator_.get_params()['solver'])

LR_train_acc = round(best_clf.score(X_train, y_train) * 100, 2)
LR_test_acc = round(best_clf.score(X_test, y_test) * 100, 2)
y_pred = best_clf.predict(X_test)
print(LR_train_acc)
print(LR_test_acc)

"""## Testing Prediciton"""

def test_name(x):
    str(x)
    new = char_v.transform([x])
    y_pred = best_clf.predict(new)
    if (y_pred == 1):
        print("This name most likely a male name")
    else:
        print("This name most likely a female name")

test_name('Virel')

test_name('Linda')

