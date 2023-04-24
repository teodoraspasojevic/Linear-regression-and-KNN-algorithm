import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# 1. PROBLEM STATEMENT AND READING DATA

pd.set_option('display.max_columns', 13)
pd.set_option('display.width', None)
data = pd.read_csv('fuel_consumption.txt')
print('Prvih pet redova data seta:')
print(data.head())

# 2. DATA ANALYSIS

print('Informacije o data setu:')
print(data.info())                                    # data profiling - only collumns EngineSize, Transmission, FuelType
                                                      # have NaN elements
print('Statistike numerickih atributa data seta:')
print(data.describe())                                # feature statistics - only for numerical features
print('Statistike nenumerickih atributa data seta:')
print(data.describe(include=[object]))                # feature statistics - only for non-numerical features

# 3. GRAPHICAL REPRESENTATION

data_numerical = data[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
                       'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]
data_nonnumerical = data[['MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE']]

plt.figure()
sb.heatmap(data_numerical.corr(), annot=True, square=True, fmt='.2f')
plt.title('Zavisnost kontinualnih atributa prikazana kroz korelacionu matricu')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od godine proizvodnje automobila')
plt.xlabel('MODELYEAR')
plt.ylabel('CO2 EMISSION')
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'MODELYEAR'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od velicine motora')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2 EMISSION')
plt.xlim(data.ENGINESIZE.min(), data.ENGINESIZE.max())
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'ENGINESIZE'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od broja cilindara')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2 EMISSION')
plt.xlim(data.CYLINDERS.min(), data.CYLINDERS.max())
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'CYLINDERS'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od potrosnje goriva u gradu')
plt.xlabel('FUELCONSUMPTION_CITY')
plt.ylabel('CO2 EMISSION')
plt.xlim(data.FUELCONSUMPTION_CITY.min(), data.FUELCONSUMPTION_CITY.max())
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'FUELCONSUMPTION_CITY'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od potrosnje goriva na otvorenom putu')
plt.xlabel('FUELCONSUMPTION_HWY')
plt.ylabel('CO2 EMISSION')
plt.xlim(data.FUELCONSUMPTION_HWY.min(), data.FUELCONSUMPTION_HWY.max())
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'FUELCONSUMPTION_HWY'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od kombinovane potrosnje goriva')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2 EMISSION')
plt.xlim(data.FUELCONSUMPTION_COMB.min(), data.FUELCONSUMPTION_COMB.max())
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'FUELCONSUMPTION_COMB'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
plt.grid()
plt.title('Zavisnost emisije CO2 od kombinovane potrosnje goriva (druga jedinica)')
plt.xlabel('FUELCONSUMPTION_COMB_MPG')
plt.ylabel('CO2 EMISSION')
plt.xlim(data.FUELCONSUMPTION_COMB_MPG.min(), data.FUELCONSUMPTION_COMB_MPG.max())
plt.ylim(data.CO2EMISSIONS.min(), data.CO2EMISSIONS.max())
plt.plot(data.loc[:, 'FUELCONSUMPTION_COMB_MPG'], data.loc[:, 'CO2EMISSIONS'], 'ro')
plt.show()

plt.figure()
g1 = sb.catplot(data=data, kind='bar', x='MAKE', y='CO2EMISSIONS', errorbar=None, estimator='mean')
g1.set_axis_labels('MAKE', 'CO2EMISSIONS')
plt.show()

plt.figure()
g2 = sb.catplot(data=data, kind='bar', x='MODEL', y='CO2EMISSIONS', errorbar=None, estimator='mean')
g2.set_axis_labels('MODEL', 'CO2EMISSIONS')
plt.show()

plt.figure()
g3 = sb.catplot(data=data, kind='bar', x='VEHICLECLASS', y='CO2EMISSIONS', errorbar=None, estimator='mean')
g3.set_axis_labels('VEHICLECLASS', 'CO2EMISSIONS')
plt.show()

plt.figure()
g4 = sb.catplot(data=data, kind='bar', x='TRANSMISSION', y='CO2EMISSIONS', errorbar=None, estimator='mean')
g4.set_axis_labels('TRANSMISSION', 'CO2EMISSIONS')
plt.show()

plt.figure()
g5 = sb.catplot(data=data, kind='bar', x='FUELTYPE', y='CO2EMISSIONS', errorbar=None, estimator='mean')
g5.set_axis_labels('FUELTYPE', 'CO2EMISSIONS')
plt.show()

# 4. DATA CLEANSING

data.ENGINESIZE.fillna(data.ENGINESIZE.mean(), inplace=True)
data.TRANSMISSION.fillna(data.TRANSMISSION.mode()[0], inplace=True)
data.FUELTYPE.fillna(data.FUELTYPE.mode()[0], inplace=True)
print('Ponovni ispis informacija o data setu da bismo proverili da li smo uspesno popunili sve NaN vrednosti: ')
print(data.info())
# deleting rows with NaN value:
# data.where(data.FUELTYPE.notnull(), inplace=True)

# 5. FEATURE ENGINEERING

# useful features:  ENGINESIZE, FUELCONSUMPTION_COMB_MPG, FUELTYPE
# useless features: MODELYEAR, CYLINDERS, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB,
#                   MAKE, MODEL, VEHICLECLASS, TRANSMISSION

data_train = data[['ENGINESIZE', 'FUELCONSUMPTION_COMB', 'FUELTYPE']]         # DATAFRAME
label = data['CO2EMISSIONS']                                                  # SERIES
print('Prvih pet redova trening skupa:')
print(data_train.head())
print('Prvih pet redova zeljenih izlaza:')
print(label.head())

ohe = OneHotEncoder(dtype=int, sparse_output=False)
fuel_type = ohe.fit_transform(data_train.FUELTYPE.to_numpy().reshape(-1, 1))
data_train = data_train.drop(columns=['FUELTYPE'])
data_train = data_train.join(pd.DataFrame(data=fuel_type, columns=ohe.get_feature_names_out(['FUELTYPE'])))
print('Prvih pet redova data seta posle kodiranja atributa FUELTYPE:')
print(data_train.head())

data_train.ENGINESIZE = data_train.ENGINESIZE/9
data_train.FUELCONSUMPTION_COMB = data_train.FUELCONSUMPTION_COMB/30
label = label/500

# 6. MODEL TRAINING

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(data_train, label, train_size=0.7, random_state=123, shuffle=False)
model.fit(X_train, y_train)
print('Parametri modela su: ', model.coef_)
print('Greska modela je: {: .2f}' .format((1-model.score(X_test, y_test))*100), '%')
print('Preciznost modela je: {: .2f}' .format((model.score(X_test, y_test))*100), '%')

# 7. MODEL TESTING

y_predicted = model.predict(X_test)
y_predicted_series = pd.Series(data=y_predicted, name='PREDICTED_CO2EMISSIONS', index=X_test.index)
data_with_prediction = pd.concat([X_test, y_test, y_predicted_series], axis=1)
print('Prvih pet redova test data seta, sa kolonom u kojoj se nalaze predikcije naseg Linear Regression modela:')
print(data_with_prediction.head())

mse_test = np.mean((y_predicted - y_test)**2)
print('Srednje kvadratna greska modela na test skupu je: {: .2f}' .format(mse_test*100), '%')

# 8. MY MODEL

class myLinearRegression():
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = np.dot(self.features, self.coeff)
        cost = np.sum((predicted - self.target)**2) / 2 / len(self.features)
        return cost

    def gradient_function(self, learning_rate):
        predicted = np.dot(self.features, self.coeff)
        error = predicted - self.target
        gradient = np.dot(self.features.T, error) / len(self.features)
        self.coeff -= learning_rate * gradient
        cost = self.cost()
        return self.coeff, cost

    def gradient_descent(self, learning_rate, num_iterations=1000):
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_function(learning_rate)
        self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        predicted = np.dot(features, self.coeff)
        predicted = predicted.reshape(-1, 1).flatten()
        return predicted

    def fit(self, features, target):
        coeff_num = features.shape[1]
        self.coeff = np.zeros(shape=coeff_num + 1).reshape(-1, 1)
        self.features = features.copy(deep=True)
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        self.features.to_numpy()
        self.target = target.to_numpy().reshape(-1, 1)              # da shape ne bi ostala torka


my_model = myLinearRegression()
my_model.fit(X_train, y_train)
learning_rates = np.array([[1], [1], [1], [0.001], [0.001], [0.001], [0.001]])
res_coeff, mse_history = my_model.gradient_descent(learning_rates, 50)
print('Parametri modela su: ', res_coeff)

y_predicted = my_model.predict(X_test)
y_predicted_series = pd.Series(data=y_predicted, name='PREDICTED_CO2EMISSIONS', index=X_test.index)
data_with_prediction = pd.concat([X_test, y_test, y_predicted_series], axis=1)
print('Prvih pet redova test data seta, sa kolonom u kojoj se nalaze predikcije naseg Linear Regression modela:')
print(data_with_prediction.head())

mse_test = np.mean((y_predicted - y_test)**2)
print('Srednje kvadratna greska modela na test skupu je: {: .2f}' .format(mse_test*100), '%')
