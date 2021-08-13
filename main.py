import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image

st.title('My first application')
# image = Image.open('img.png')
# st.image(image, use_column_width=True, width=100)

# side bar for dataset selection
dataset_sel = st.sidebar.selectbox('Select dataset', ('Breast cancer', 'Iris', 'Wine'))
# Algorithm selection
algo_sel = st.sidebar.selectbox('Select ML Algorithm', ('KNN', 'SVM'))


def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y


x, y = get_dataset(dataset_sel)

st.dataframe(x)
st.write('Shape of dataset:', x.shape)
st.write('Unique target variables:', len(np.unique(y)))

fig = plt.figure()
sns.boxplot(data=x, orient='h')
st.pyplot(fig)


# get the parameters for the algorithm
def get_params(algo_name):
    params = dict()
    if algo_name == 'KNN':
        params['k'] = st.sidebar.slider('k', 1, 15)
    elif algo_name == 'SVM':
        params['C'] = st.sidebar.slider('C', 0.01, 10.0)
    return params


params = get_params(algo_sel)
# Accessing classifier


def get_classifier(algo_name, params):
    if algo_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    return clf


clf = get_classifier(algo_sel, params)

# Train test split of dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

st.write('Confusion Matrix')
cm = metrics.confusion_matrix(y_test, y_pred)
fig = plt.figure()
sns.heatmap(cm, annot=True)
st.pyplot(fig)
st.write('Accuracy:', metrics.accuracy_score(y_test, y_pred))