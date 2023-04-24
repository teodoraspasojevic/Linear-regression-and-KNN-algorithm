# Linear-regression-and-KNN-algorithm
In this assignment, we will implement linear regression and KNN algorithm and apply it on two given datasets: cakes and fuel_consumption. The cakes dataset consists of the amounts of different ingredients needed to make cupcakes or muffins. The fuel_consumption dataset contains different numerical and non-numerical information about various cars and their effect on CO2 emissions.

**Data Cleansing and Feature Engineering**

Before applying an algorithm for prediction or classification, we need to analyze the given datasets, clean them, and choose the relevant features for further use. To analyze the datasets, we will use the info and describe functions from the pandas library to obtain different statistics. We will also plot the dependency of the desired output on the values of different features and use a heatmap to visualize the correlations between different features. Data cleansing involves removing NaN values from the datasets, which is only needed for the fuel_consumption dataset. Additionally, we need to code non-numerical features into numerical ones, for which we will use the LabelEncoder and OneHotEncoder from the scikit-learn library. Once we've completed all of the above, we can choose the features that have the most significant impact on the desired output.

**Model Training and Testing**

After selecting the features to use in the implementation, we will split the data into a training set and a test set. We will train our model on the training set and estimate the error of the model on the test set.

**KNN Algorithm**

We will use the KNN algorithm to predict whether a given combination of ingredients is used for making a cupcake or a muffin. Since the output of our algorithm can be either one of two classes, we call this a classification problem. The KNN algorithm works by finding the K nearest neighbors of the current combination of input features and checking if there are more neighbors that belong to class 1 or class 2. Based on this information, the algorithm makes its prediction.

**Linear Regression**

We will use linear regression to predict the CO2 emission of a specific car by analyzing different features of the given car. Since the output of the algorithm is a specific value that represents the estimated CO2 emission, we call this a prediction problem. The linear regression algorithm works by estimating the parameters of a linear function that best fits the trend of the output value based on the values of the input features. The parameters are estimated on the training set, fixed, and used to predict the output for input features from the test set.






