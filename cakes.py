import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seaborn import catplot, heatmap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from queue import PriorityQueue

# 1. PROBLEM STATEMENT AND READING DATA

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)
data = pd.read_csv('cakes.txt')
print('Prvih pet redova data seta:')
print(data.head())

# 2. DATA ANALYSIS

print('Informacije o data setu:')
print(data.info())                                   # data profiling
print('Statistike numerickih atributa data seta:')
print(data.describe())                               # feature statistics -  only for numerical features
print('Statistike nenumerickih atributa data seta:')
print(data.describe(include=[object]))               # feature statistics -  only for non-numerical features

# 3. GRAPHICAL REPRESENTATION

data_numerical = data[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]

plt.figure()
heatmap(data_numerical.corr(), annot=True, square=True, fmt='.2f')
plt.title('Zavisnost kontinualnih atributa prikazana kroz korelacionu matricu')
plt.show()

plt.figure()
g1 = catplot(data=data, kind='bar', x='type', y='flour', errorbar=None, estimator='mean')
g1.set_axis_labels('type', 'flour')
plt.show()

plt.figure()
g2 = catplot(data=data, kind='bar', x='type', y='eggs', errorbar=None, estimator='mean')
g2.set_axis_labels('type', 'eggs')
plt.show()

plt.figure()
g3 = catplot(data=data, kind='bar', x='type', y='sugar', errorbar=None, estimator='mean')
g3.set_axis_labels('type', 'sugar')
plt.show()

plt.figure()
g4 = catplot(data=data, kind='bar', x='type', y='milk', errorbar=None, estimator='mean')
g4.set_axis_labels('type', 'milk')
plt.show()

plt.figure()
g5 = catplot(data=data, kind='bar', x='type', y='butter', errorbar=None, estimator='mean')
g5.set_axis_labels('type', 'butter')
plt.show()

plt.figure()
g6 = catplot(data=data, kind='bar', x='type', y='baking_powder', errorbar=None, estimator='mean')
g6.set_axis_labels('type', 'baking_powder')
plt.show()

# 4. DATA CLEANSING

# no NaN values

# 5. FEATURE ENGINEERING

data_cupcake = data.loc[data.type == 'cupcake']
print('Statistike podataka koji su oznaceni kao cupcake:')
print(data_cupcake.describe())

data_muffin = data.loc[data.type == 'muffin']
print('Statistike podataka koji su oznaceni kao muffin:')
print(data_muffin.describe())

le = LabelEncoder()
data.type = le.fit_transform(data.type)
print('Prvih pet redova data seta posle kodiranja atributa type:')
print(data.head())

# If we look at the graphs or isolate the data on cupcakes and muffins and examine which attributes have small/large
# differences in the mean values, we can determine which characteristics are relevant to us:
# => Useful features: sugar, butter, eggs, flour
# => Useless features: baking powder, milk

data_train = data[['sugar', 'butter', 'eggs', 'flour']]  # DataFrame
label = data['type']                                     # Series
print('Prvih pet redova trening skupa:')
print(data_train.head())
print('Prvih pet redova zeljenih izlaza:')
print(label.head())

# 6. MODEL TRAINING

model = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(data_train, label, train_size=0.8, random_state=123, shuffle=True)
model.fit(X_train, y_train)
print('Parametri modela su: ', model.get_params())
print('Greska modela je: {: .2f}' .format((1-model.score(X_test, y_test))*100), '%')
print('Preciznost modela je: {: .2f}' .format((model.score(X_test, y_test))*100), '%')

# 7. MODEL TESTING

y_predicted = model.predict(X_test)
y_predicted_series = pd.Series(data=y_predicted, name='predicted_type', index=X_test.index)
data_with_prediction = pd.concat([X_test, y_test, y_predicted_series], axis=1)
print('Prvih pet redova test data seta, sa kolonom u kojoj se nalaze predikcije naseg KNN modela:')
print(data_with_prediction.head())

mse_test = np.mean((y_predicted - y_test)**2)
print('Srednje kvadratna greska modela na test skupu je: {: .2f}' .format(mse_test*100), '%')

# 8. MY MODEL

class myKNeighboursClassifier():
    def __int__(self):
        self.features = None
        self.targets = None
        self.k = None

    def fit(self, features, targets, k):
        self.features = features.copy(deep=True)
        self.features = self.features.to_numpy()
        self.targets = targets.to_numpy().reshape(-1, 1)  # da shape ne bi ostala torka
        self.k = k

    def predict(self, features):
        m1 = self.features.shape[0]
        n1 = self.features.shape[1]
        m2 = features.shape[0]
        predictions = np.zeros(m2)
        features = features.to_numpy()

        for i in range(m2):
            distances = PriorityQueue()
            for j in range(m1):
                distance = 0
                for k in range(n1):
                    distance += np.sqrt((self.features[j][k] - features[i][k])**2)
                distances.put([distance, j])
            cupcake_number = 0
            muffin_number = 0
            neighbours_number = self.k
            for l in range(neighbours_number):
                tmp = distances.get()
                min_index = tmp[1]
                min_label = self.targets[min_index]
                if min_label == 0:
                    cupcake_number += 1
                else:
                    muffin_number += 1
            if cupcake_number >= muffin_number:
                predictions[i] = 0
            else:
                predictions[i] = 1
        return predictions

    def score(self, features, targets):
        targets = targets.to_numpy()
        predicted = self.predict(features)
        correct = 0
        for i in range(len(targets)):
           if predicted[i] == targets[i]:
                correct += 1
        return correct/len(targets)


my_model = myKNeighboursClassifier()
my_model.fit(X_train, y_train, 5)

y_predicted = my_model.predict(X_test)
y_predicted_series = pd.Series(data=y_predicted, name='predicted_type', index=X_test.index)
data_with_prediction = pd.concat([X_test, y_test, y_predicted_series], axis=1)
print('Prvih pet redova test data seta, sa kolonom u kojoj se nalaze predikcije naseg KNN modela:')
print(data_with_prediction.head())
print('Greska modela je: {: .2f}' .format((1 - my_model.score(X_test, y_test))*100), '%')
print('Preciznost modela je: {: .2f}' .format((my_model.score(X_test, y_test))*100), '%')

mse_test = np.mean((y_predicted - y_test)**2)
print('Srednje kvadratna greska modela na test skupu je: {: .2f}' .format(mse_test*100), '%')
