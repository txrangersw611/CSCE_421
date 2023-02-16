################
################
## Q1
################
################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from typing import Tuple, List
import scipy.stats


# Download and read the data.
def read_data(filename: str) -> pd.DataFrame:
    '''
        read train data and return dataframe
    '''
    data = pd.read_csv(filename)

    return data

# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    #remove NaN values for both
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    #split the dataframes
    train_data = df_train['x']
    train_label = df_train['y']

    test_data = df_test['x']
    test_label = df_test['y']

    return train_data, train_label, test_data, test_label

# Implement LinearRegression class
class LinearRegression_Local:   #give up on this
    def __init__(self, learning_rate=0.00001, iterations=30):        
        self.learning_rate = learning_rate
        self.iterations    = iterations

    # Function for model training         
    def fit(self, X, Y):
        # weight initialization, W will be the weights multiplied by X 
        self.w1 = 1
        self.w0 = 0

        # data
        self.x = X
        self.y = Y

        
        # gradient descent learning 
        for i in range(self.iterations):
            self.update_weights()

        return self

    # Helper function to update weights in gradient descent      
    def update_weights(self):
        # predict on data and calculate gradients 
        num = self.x.shape[0]
        y = self.predict()
        y_err = np.array(num)

        #do RSS to see how off the predictions are from Y
        for i in range(num):
            y_err[i] = (self.y[i] - y[i])**2

        #adjust weights based on the y_err


        return self

    # Hypothetical function  h( x )       
    def predict(self, X):
        y = self.w1*(self.x) + self.w0

        return y

# Build your model
def build_model(train_x: np.array, train_y: np.array):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    reg = LinearRegression() #have to use the local one at some point

    train_x = np.expand_dims(train_x, axis=1)
    # train_y = np.expand_dims(train_y, axis=1)

    # print(train_x.shape)
    # print(train_y.shape)

    reg.fit(train_x, train_y)

    return reg

# Make predictions with test set
def pred_func(model, X_test):
    '''
        return numpy array comprising of prediction on test set using the model
    '''
    X_test = np.expand_dims(X_test, axis=1)
    #print(X_test.shape)

    predictions = model.predict(X_test)
    return predictions

# Calculate and print the mean square error of your prediction
def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
    '''
    #print(y_test.shape[0])
    meanSqEr = 0.0
    for i in range(y_test.shape[0]):
        meanSqEr += (y_test[i]-pred[i])**2
    
    meanSqEr /= y_test.shape[0]
    #print(meanSqEr)
    return meanSqEr

# ################
# ################
# ## Q2
# ################
# ################

# Download and read the data.
def read_training_data(filename: str) -> tuple:
    '''
        read train data into a dataframe df1, store the top 10 entries of the dataframe in df2
        and return a tuple of the form (df1, df2, shape of df1)   
    '''
    df1 = pd.read_csv(filename)
    df2 = df1.iloc[:10]

    # print(df2.head(11))
    # print(df2.shape)

    return df1, df2, df1.shape

# Prepare your input data and labels
def data_clean(df_train: pd.DataFrame) -> tuple:
    '''
        check for any missing values in the data and store the missing values in series s, drop the entries corresponding 
        to the missing values and store dataframe in df_train and return a tuple in the form: (s, df_train)
    '''
    s = df_train.isnull().sum()
    df_train = df_train.dropna()

    return s, df_train

def feature_extract(df_train: pd.DataFrame) -> tuple:
    '''
        New League is the label column.
        Separate the data from labels.
        return a tuple of the form: (features(dtype: pandas.core.frame.DataFrame), label(dtype: pandas.core.series.Series))
    '''
    label = df_train.loc[:, "NewLeague"]
    #print(type(label))

    features = df_train.drop("NewLeague", axis='columns')
    #print(type(features))

    return features, label

def data_preprocess(feature: pd.DataFrame) -> pd.DataFrame:
    '''
        Separate numerical columns from nonnumerical columns. (In pandas, check out .select dtypes(exclude = ['int64', 'float64']) and .select dtypes(
        include = ['int64', 'float64']). Afterwards, use get dummies for transforming to categorical. Then concat both parts (pd.concat()).
        and return the concatenated dataframe.
    '''
    numerical = feature.select_dtypes(include = ['int64', 'float64'])
    categorical = feature.select_dtypes(exclude = ['int64', 'float64'])
    newCategorical = pd.get_dummies(categorical)

    combined = pd.concat([newCategorical, numerical], axis=1)
    return combined


def label_transform(labels: pd.Series) -> pd.Series:
    '''
        Transform the labels into numerical format and return the labels
    '''
    labels = labels.replace({'A':0})
    labels = labels.replace({'N':1})
    return labels

# ################
# ################
# ## Q3
# ################
# ################ 
def data_split(features: pd.DataFrame, label:pd.Series, random_state  = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Split 80% of data as a training set and the remaining 20% of the data as testing set using the given random state
        return training and testing sets in the following order: X_train, X_test, y_train, y_test
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size = 0.2, random_state=random_state)

    return X_train, X_test, Y_train, Y_test

def train_linear_regression( x_train: np.ndarray, y_train:np.ndarray):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    reg = LinearRegression() #have to use the local one at some point

    # x_train = np.expand_dims(x_train, -1)
    # y_train = np.expand_dims(y_train, -1)
    # print(x_train.shape)
    # print(y_train.shape)


    reg.fit(x_train, y_train)

    return reg

def train_logistic_regression( x_train: np.ndarray, y_train:np.ndarray, max_iter=1000000):
    '''
        Instantiate an object of LogisticRegression class, train the model object
        use provided max_iterations for training logistic model
        using training data and return the model object
    '''
    reg = LogisticRegression(max_iter=max_iter) 

    # print(x_train.shape)
    # print(y_train.shape)

    reg.fit(x_train, y_train)

    return reg

def models_coefficients(linear_model, logistic_model) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return the tuple consisting the coefficients for each feature for Linear Regression 
        and Logistic Regression Models respectively
    '''
    linReg = np.array(linear_model.coef_)

    logReg = np.array(logistic_model.coef_)

    return linReg, logReg


def linear_pred_and_area_under_curve(linear_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    linear_reg_pred = linear_model.predict(x_test)

    linear_reg_fpr, linear_reg_tpr, linear_threshold = metrics.roc_curve(y_test, linear_reg_pred)
    linear_reg_area_under_curve = metrics.roc_auc_score(y_test, linear_reg_pred)

    #logistic_reg_pred = metrics.precision_score(y_test, logistic_reg_pred)

    return linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve



def logistic_pred_and_area_under_curve(logistic_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [log_reg_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    logistic_reg_pred_temp = logistic_model.predict_proba(x_test)
    logistic_reg_pred = logistic_reg_pred_temp[:,1]

    logistic_reg_fpr, logistic_reg_tpr, logistic_threshold = metrics.roc_curve(y_test, logistic_reg_pred)
    # print(logistic_threshold)
    # print(logistic_reg_tpr)
    logistic_reg_area_under_curve = metrics.roc_auc_score(y_test, logistic_reg_pred)


    return logistic_reg_pred, logistic_reg_fpr, logistic_reg_tpr, logistic_threshold, logistic_reg_area_under_curve



def optimal_thresholds(linear_threshold: np.ndarray, linear_reg_fpr: np.ndarray, linear_reg_tpr: np.ndarray, log_threshold: np.ndarray, log_reg_fpr: np.ndarray, log_reg_tpr: np.ndarray) -> Tuple[float, float]:
    '''
        return the tuple consisting the thresholds of Linear Regression and Logistic Regression Models respectively
    '''
    linmax = np.argmax(linear_reg_tpr-linear_reg_fpr)
    logmax = np.argmax(log_reg_tpr-log_reg_fpr)

    return linear_threshold[linmax], log_threshold[logmax]
    

def stratified_k_fold_cross_validation(num_of_folds: int, shuffle: True, features: pd.DataFrame, label: pd.Series):
    '''
        split the data into 5 groups. Checkout StratifiedKFold in scikit-learn
    '''
    skt = StratifiedKFold(n_splits=num_of_folds, shuffle=shuffle, random_state=None)

    # print(features.shape)
    # print(label.shape)

    return skt

def train_test_folds(skf, num_of_folds: int, features: pd.DataFrame, label: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    '''
        train and test in for loop with different training and test sets obatined from skf. 
        use a PENALTY of 12 for logitic regression model for training
        find features in each fold and store them in features_count array.
        populate auc_log and auc_linear arrays with roc_auc_score of each set trained on logistic regression and linear regression models respectively.
        populate f1_dict['log_reg'] and f1_dict['linear_reg'] arrays with f1_score of trained logistic and linear regression models on each set
        return features_count, auc_log, auc_linear, f1_dict dictionary
    '''
    features_count = np.array([])
    auc_log = np.array([])
    auc_linear = np.array([])
    f1_dict = {'log_reg':[], 'linear_reg':[]}


    #Create linear and logistic regression variables to be used to get score and f1 values
    lin_reg = LinearRegression()
    log_reg = LogisticRegression(penalty='l2', max_iter=1000000)

    #This loop will run 5 times to test each of the 5 split data sets
    for train_index, test_index in skf.split(features, label):
        #Get the data from the current section
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]
        label_train, label_test = label.iloc[train_index].values.ravel(), label.iloc[test_index].values.ravel()


        #Store features of this data section in the features_count array
        features_count = np.append(features_count, features_train.shape[1])

        #The current data is then used to fit a linear model and get the 
        #Fit the data
        lin_reg.fit(features_train, label_train)

        #Get predictions
        label_preds_lin = lin_reg.predict(features_test).round().clip(0).astype(int)

        #Get the ROC_AUC score
        auc_linear = np.append(auc_linear, metrics.roc_auc_score(label_test, label_preds_lin))

        # print(label_preds_lin)
        # print(label_test)

        #Get F1 score
        f1_dict['linear_reg'].append(metrics.f1_score(label_test, label_preds_lin))

        #The current data is then used to fit a logistic model and get the 
        #Fit the data
        log_reg.fit(features_train, label_train)

        #Get predictions
        label_preds_log = log_reg.predict(features_test).round().clip(0).astype(int)
        # label_preds_log = label_preds_log_temp[:,1]

        #Get the ROC_AUC score
        auc_log = np.append(auc_log, metrics.roc_auc_score(label_test, label_preds_log))

        #Get F1 score
        f1_dict['log_reg'].append(metrics.f1_score(label_test, label_preds_log))


    return features_count, auc_log, auc_linear, f1_dict
        
    


def is_features_count_changed(features_count: np.array) -> bool:
    '''
        compare number of features in each fold (features_count array's each element)
        return true if features count doesn't change in each fold. else return false
    '''
    # print(features_count.size())

    # for i in range(features.shape[0]-1):
    #     print(features_count[i])

    if (features_count[0] != features_count[1] != features_count[2] != features_count[3] != features_count[4]):
        return False

    return True


def mean_confidence_interval(data: np.array, confidence=0.95) -> Tuple[float, float, float]:
    '''
        To calculate mean and confidence interval, in scipy checkout .sem to find standard error of the mean of given data (AUROCs/ f1 
        scores of each model, linear and logistic trained on all sets). 
        Then compute Percent Point Function available in scipy and mutiply it with standard error calculated earlier to calculate h. 
        The required interval is from mean-h to mean+h
        return the tuple consisting of mean, mean -h, mean+h
    '''
    st_error = scipy.stats.sem(data)
    h = st_error * scipy.stats.t.ppf((1+confidence)/2, len(data)-1)
    mean = np.mean(data)

    return mean, mean-h, mean+h

if __name__ == "__main__":

    ################
    ################
    ## Q1
    ################
    ################
    data_path_train   = "LinearRegression/train.csv"
    data_path_test    = "LinearRegression/test.csv"
    df_train, df_test = read_data(data_path_train), read_data(data_path_test)

    # print(df_train.head())
    # print(df_test.head())

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)

    model = build_model(train_X, train_y)

    # Make prediction with test set
    preds = pred_func(model, test_X)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds)

    # plot your prediction and labels, you can save the plot and add in the report

    plt.plot(test_y, label='label')
    plt.plot(preds, label='pred')
    plt.legend()
    plt.show()


#     ################
#     ################
#     ## Q2
#     ################
#     ################

    data_path_training   = "Hitters.csv"


    train_df, df2, df_train_shape = read_training_data(data_path_training)
    #print(train_df.head())
    #print(df2.head())
    s, df_train_mod = data_clean(train_df)
    #print(df_train_mod.head())
    features, label = feature_extract(df_train_mod)
    final_features  = data_preprocess(features)
    #print(label)
    final_label     = label_transform(label)
    #print(final_label)
#     ################
#     ################
#     ## Q3
#     ################
#     ################

    num_of_folds = 5
    max_iter = 100000008
    X = final_features
    y = final_features
    auc_log = []
    auc_linear = []
    features_count = []
    f1_dict = {'log_reg': [], 'linear_reg': []}

    X_train, X_test, y_train, y_test = data_split(final_features, final_label)

    linear_model = train_linear_regression(X_train, y_train)

    logistic_model = train_logistic_regression(X_train, y_train)

    linear_coef, logistic_coef = models_coefficients(linear_model, logistic_model)

    # print(linear_coef)
    # print(logistic_coef)

    linear_y_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve = linear_pred_and_area_under_curve(linear_model, X_test, y_test)

    log_y_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve = logistic_pred_and_area_under_curve(logistic_model, X_test, y_test)

    # print(linear_reg_area_under_curve)
    # print(log_reg_area_under_curve)

    # print(linear_y_pred)
    # print(log_y_pred)

    plt.plot(log_reg_fpr, log_reg_tpr, label='logistic')
    plt.plot(linear_reg_fpr, linear_reg_tpr, label='linear')
    plt.legend()
    plt.show()

    linear_optimal_threshold, log_optimal_threshold = optimal_thresholds(linear_threshold, linear_reg_fpr, linear_reg_tpr, log_threshold, log_reg_fpr, log_reg_tpr)

    #print(linear_optimal_threshold, log_optimal_threshold)

    skf = stratified_k_fold_cross_validation(num_of_folds, True, final_features, final_label)

    features_count, auc_log, auc_linear, f1_dict = train_test_folds(skf, num_of_folds, final_features, final_label)

    print("Does features change in each fold?")

    # call is_features_count_changed function and return true if features count changes in each fold. else return false
    value = is_features_count_changed(features_count)
    
    if (value):
        print("features_count changes")
    else:
        print("features_count did not change")

    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = 0, 0, 0
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = 0, 0, 0

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_interval = 0, 0, 0
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = 0, 0, 0


    #Find mean and 95% confidence interval for the AUROCs for each model and populate the above variables accordingly
    #Hint: use mean_confidence_interval function and pass roc_auc_scores of each fold for both models (ex: auc_log)
    #Find mean and 95% confidence interval for the f1 score for each model.

    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = mean_confidence_interval(auc_linear)
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = mean_confidence_interval(auc_log)

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_intervel = mean_confidence_interval(f1_dict['linear_reg'])
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = mean_confidence_interval(f1_dict['log_reg']) 

    print()
    print("The mean for linear regression is", auc_linear_mean)
    print("The confidence interval of linear regression is from", auc_linear_open_interval, "to", auc_linear_close_interval)
    print("The mean for logarithmic regression is", auc_log_mean)
    print("The confidence interval for logarithmic regression is from", auc_log_open_interval, "to", auc_log_close_interval)
    print()
    print("The mean for linear regression is", f1_linear_mean)
    print("The confidence interval of linear regression is from", f1_linear_open_interval, "to", f1_linear_close_interval)
    print("The mean for logarithmic regression is", f1_log_mean)
    print("The confidence interval for logarithmic regression is from", f1_log_open_interval, "to", f1_log_close_interval)


