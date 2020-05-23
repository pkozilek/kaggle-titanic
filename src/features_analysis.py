import plotly as plt
import numpy as np
import pandas as pd
import functions.graphs as g

features = pd.read_csv('datasets/features.csv', index_col=0)
features_survived = features.loc[features.Survived == 1]
features_died = features.loc[features.Survived == 0]

# Age bias
age_bias = g.histogram(
    data_list=[features_survived.Age, features_died.Age],
    labels=['Survived', 'Died'],
    title='Normalized age bias analysis',
    x_bins=[-1, 100, 5],
    histnorm='percent',
    xaxis_title='Age',
    yaxis_title='%'
)
age_bias.show()

# Sex bias
sex_bias = g.histogram(
    data_list=[features_survived.Sex, features_died.Sex],
    labels=['Survived', 'Died'],
    title='Normalized sex bias analysis',
    x_bins=[1, 3, 1],
    histnorm='percent',
    xaxis_title='Sex',
    yaxis_title='%'
)
sex_bias.show()

# Pclass bias
pclass = g.histogram(
    data_list=[features_survived.Pclass, features_died.Pclass],
    labels=['Survived', 'Died'],
    title='Normalized Pclass bias analysis',
    x_bins=[0, 20, 1],
    histnorm='percent',
    xaxis_title='Pclass',
    yaxis_title='%'
)
pclass.show()

# SibSp bias
sibsp = g.histogram(
    data_list=[features_survived.SibSp, features_died.SibSp],
    labels=['Survived', 'Died'],
    title='Normalized SibSp bias analysis',
    x_bins=[0, 9, 1],
    histnorm='percent',
    xaxis_title='SibSp',
    yaxis_title='%'
)
sibsp.show()

# Parch bias
parch = g.histogram(
    data_list=[features_survived.Parch, features_died.Parch],
    labels=['Survived', 'Died'],
    title='Normalized Parch bias analysis',
    x_bins=[0, 7, 1],
    histnorm='percent',
    xaxis_title='Parch',
    yaxis_title='%'
)
parch.show()

# Embarked bias
embarked = g.histogram(
    data_list=[features_survived.Embarked, features_died.Embarked],
    labels=['Survived', 'Died'],
    title='Normalized Embarked bias analysis',
    x_bins=[0, 4, 1],
    histnorm='percent',
    xaxis_title='Embarked',
    yaxis_title='%'
)
embarked.show()

# Fare bias
fare = g.histogram(
    data_list=[features_survived.Fare, features_died.Fare],
    labels=['Survived', 'Died'],
    title='Normalized Fare bias analysis',
    x_bins=[0, 100, 5],
    histnorm='percent',
    xaxis_title='Fare',
    yaxis_title='%'
)
fare.show()