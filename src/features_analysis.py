import plotly as plt
import numpy as np
import pandas as pd
import functions.graphs as g

features = pd.read_csv('datasets/features.csv', index_col=0)
features_survived = features.loc[features.Survived == 1]
features_died = features.loc[features.Survived == 0]
features_ac = features.loc[features.Age != 0] # Age completed
features_ac_survived = features_ac.loc[features_ac.Survived == 1]
features__ac_died = features_ac.loc[features_ac.Survived == 0]

### Bias Analysis

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

# age_bias.show()
# sex_bias.show()
# pclass.show()
# sibsp.show()
# parch.show()
# embarked.show()
# fare.show()

### Age correlation analysis

age_sex_scatter = g.scatterplot(
    x=[features_ac_survived.Age, features__ac_died.Age],
    y=[features_ac_survived.Sex, features__ac_died.Sex],
    labels=['Survived', 'Died'],
    x_jitter=0.5,
    y_jitter=0.25,
    title='Sex x Age',
    xaxis_title='Age',
    yaxis_title='Sex'
)

age_pclass_scatter = g.scatterplot(
    x=[features_ac_survived.Age, features__ac_died.Age],
    y=[features_ac_survived.Pclass, features__ac_died.Pclass],
    labels=['Survived', 'Died'],
    x_jitter=0.5,
    y_jitter=0.25,
    title='Pclass x Age',
    xaxis_title='Age',
    yaxis_title='Pclass'
)

age_sibsp_scatter = g.scatterplot(
    x=[features_ac_survived.Age, features__ac_died.Age],
    y=[features_ac_survived.SibSp, features__ac_died.SibSp],
    labels=['Survived', 'Died'],
    x_jitter=0.5,
    y_jitter=0.25,
    title='SibSp x Age',
    xaxis_title='Age',
    yaxis_title='SibSp'
)

age_parch_scatter = g.scatterplot(
    x=[features_ac_survived.Age, features__ac_died.Age],
    y=[features_ac_survived.Parch, features__ac_died.Parch],
    labels=['Survived', 'Died'],
    x_jitter=0.5,
    y_jitter=0.25,
    title='Parch x Age',
    xaxis_title='Age',
    yaxis_title='Parch'
)

age_embarked_scatter = g.scatterplot(
    x=[features_ac_survived.Age, features__ac_died.Age],
    y=[features_ac_survived.Embarked, features__ac_died.Embarked],
    labels=['Survived', 'Died'],
    x_jitter=0.5,
    y_jitter=0.25,
    title='Embarked x Age',
    xaxis_title='Age',
    yaxis_title='Embarked'
)

age_fare_scatter = g.scatterplot(
    x=[features_ac_survived.Age, features__ac_died.Age],
    y=[features_ac_survived.Fare, features__ac_died.Fare],
    labels=['Survived', 'Died'],
    x_jitter=0.5,
    y_jitter=0.25,
    title='Fare x Age',
    xaxis_title='Age',
    yaxis_title='Fare'
)


# age_sex_scatter.show()
# age_pclass_scatter.show()
# age_sibsp_scatter.show()
# age_parch_scatter.show()
# age_embarked_scatter.show()
# age_fare_scatter.show()