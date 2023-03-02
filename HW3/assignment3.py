#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
import random
from typeguard import typechecked

#using this for testing function
import time

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the filename. Load the data it in a dataframe and return it.
    """
    data = pd.read_csv(filename)

    return data


@typechecked
def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: Coding: Preprocessing the Data.)
    Return the final features and final label in same order
    You may use the same code you submiited for problem 2 of HW2
    """
    #clean_data function from hw2
    df = df.dropna()

    #feature_extract from hw2
    labels = df.loc[:, "NewLeague"]
    features = df.drop(["NewLeague", "Player"], axis='columns') #drop player bc instructions said not to include Player

    #data_preprocess from hw2
    numerical = features.select_dtypes(include = ['int64', 'float64'])
    categorical = features.select_dtypes(exclude = ['int64', 'float64'])
    newCategorical = pd.get_dummies(categorical)

    fixedFeatures = pd.concat([newCategorical, numerical], axis=1)

    #label_transform from hw2
    labels = labels.replace({'A':0})
    labels = labels.replace({'N':1})

    return fixedFeatures, labels



@typechecked
def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split 80% of data as a training set and the remaining 20% of the data as testing set
    return training and testing sets in the following order: X_train, X_test, y_train, y_test
    """
    X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size = test_size)

    return X_train, X_test, Y_train, Y_test


@typechecked
def train_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Ridge Regression, train the model object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {1e-3: [], 1e-2: [], 1e-1: [], 1: [], 1e1: [], 1e2: [], 1e3: []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]


    for i in range(n): #change back to n when done testing
        for j in range(len(lambda_vals)):        
            #train the model using the current lambda and get predictions
            ridgeReg = Ridge(alpha=lambda_vals[j], max_iter=max_iter)
            ridgeReg.fit(x_train, y_train)
            y_pred = ridgeReg.predict(x_test)

            #get the probability and append it to the dictionary with the correct lambda key value
            y_pred_probs = 1/ (((1+lambda_vals[j]) * (1 - y_pred)) / (1 + y_pred)) 
            aucs[lambda_vals[j]].append(roc_auc_score(y_test, y_pred_probs))
    
    
    # print(n)
    # print(len(lambda_vals))
    # print(len(aucs["ridge"]))

    print("ridge mean AUCs:")
    ridge_aucs = pd.DataFrame(aucs)
    ridge_mean_auc = {}

    #print(ridge_aucs)
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean()):
        ridge_mean_auc[lambda_val] = ridge_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % ridge_auc)
    return ridge_mean_auc


@typechecked
def train_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter=int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {1e-3: [], 1e-2: [], 1e-1: [], 1: [], 1e1: [], 1e2: [], 1e3: []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(n): #change back to n when done testing
        for j in range(len(lambda_vals)):        
            #train the model using the current lambda and get predictions
            lassoReg = Lasso(alpha=lambda_vals[j], max_iter=max_iter)
            lassoReg.fit(x_train, y_train)
            y_pred = lassoReg.predict(x_test)

            #get the probability and append it to the dictionary with the correct lambda key value
            y_pred_probs = 1/ (((1+lambda_vals[j]) * (1 - y_pred)) / (1 + y_pred)) 
            aucs[lambda_vals[j]].append(roc_auc_score(y_test, y_pred_probs))
    
    

    print("lasso mean AUCs:")
    lasso_mean_auc = {}
    lasso_aucs = pd.DataFrame(aucs)
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        lasso_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)
    return lasso_mean_auc


@typechecked
def ridge_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Ridge, np.ndarray]:
    """
    return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
    of the model
    """
    ridgeReg = Ridge(alpha=optimal_alpha, max_iter=max_iter)
    ridgeReg.fit(x_train, y_train)

    coeff = ridgeReg.coef_

    return ridgeReg, coeff


@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:
    """
    return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
    of the model
    """
    lassoReg = Lasso(alpha=optimal_alpha, max_iter=max_iter)
    lassoReg.fit(x_train, y_train)

    coeff = lassoReg.coef_

    return lassoReg, coeff


@typechecked
def ridge_area_under_curve(
    model_R, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of trained Ridge model used to find coefficients,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    #ripped off from linear_pred_and_area_under_curve in hw2
    ridge_reg_pred = model_R.predict(x_test)

    ridge_reg_area_under_curve = roc_auc_score(y_test, ridge_reg_pred)


    return ridge_reg_area_under_curve


@typechecked
def lasso_area_under_curve(
    model_L, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of Lasso Model,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    #ripped off from linear_pred_and_area_under_curve in hw2
    lasso_reg_pred = model_L.predict(x_test)

    lasso_reg_area_under_curve = roc_auc_score(y_test, lasso_reg_pred)


    return lasso_reg_area_under_curve


class Node:
    @typechecked
    def __init__(
        self,
        split_val: float,
        data: Any = None,
        left: Any = None,
        right: Any = None,
    ) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)

        self.left = left
        self.right = right
        self.split_val = split_val  # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.data = data  # data can be anything! we recommend dictionary with all variables you need


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth
        self.root_node = Node(0, self.data)


    #adding this to see how the tree looks after being created
    def treeChecker(self, node, depth) -> None:
        print("-------------------------")
        print("Depth: ", depth)
        print("Split Value: ", node.split_val)

        time.sleep(1)

        if (node.left != None):
            self.treeChecker(node.left, depth+1)

        if (node.right != None):
            self.treeChecker(node.right, depth+1)

    @typechecked
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        #get the initial best split for the data so root node has a value
        #self.root_node = self.get_best_split(self.data)

        # print(self.root_node.left.data.shape[0])
        # print(self.root_node.right.data.shape[0])

        #calling split will recursively call itself until the tree is built to the correct depth
        self.split(self.root_node, 1)

        # print(self.root_node.left.data.shape[0])
        # print(self.root_node.right.data.shape[0])

        return self.root_node


    @typechecked
    def mean_squared_error(
        self, left_split: np.ndarray, right_split: np.ndarray
    ) -> float:
        """
        Calculate the mean squared error for a split dataset
        left split is a list of rows of a df, rightmost element is label
        return the sum of mse of left split and right split
        """
        #check if either of the splits is empty or else np.mean gives me errors
        if len(left_split) == 0 or len(right_split) == 0:
            return 1000.0

        mse = np.mean(np.square(left_split[:, -1] - np.mean(left_split[:, -1]))) + np.mean(np.square(right_split[:, -1] - np.mean(right_split[:, -1])))

        #print(mse)

        return mse
    

    @typechecked
    def split(self, node: Node, depth: int) -> None:
        """
        Do the split operation recursively
        """
        #check if the depth is 0. This will the the stopping case
        if (depth == self.max_depth):
            return
        

        #split the data and save the resulting node into splittedNode
        tempnode = self.get_best_split(node, node.data)

        node.left = tempnode.left
        node.right = tempnode.right
        node.split_val = tempnode.split_val

        #print("Node at depth", depth, " has split and the data in its child nodes is", node.left.data.shape[0], "for the left and", node.right.data.shape[0], "for the right.")

        #increment the depth so when the next recursive call is made it has the updated depth
        depth += 1

        # print("left: ", node.left.data.shape)
        # print("right: ", node.right.data.shape)

        #call split again. This will split the left and right node until the depth is reached
        self.split(node.left, depth)
        self.split(node.right, depth)


    @typechecked
    def get_best_split(self, node: Node, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        #doing the first split to get a base score to compare as I loop
        left_best = np.empty((1, data.shape[1]))
        right_best = np.empty((1, data.shape[1]))
        split_index = 0
        feature_index = 0
        best_mse = 1000.0

        temp_mse = 0.0

        # start = 1

        #check if either the left temp or right temp have a length of 0, if so move on to next iteration
        # if (len(left_best) == 0 or len(right_best) == 0):
        #     left_best, right_best = self.one_step_split(1, data[1, 0], data)
        #     start = 2
        #     split_index = 1

        for i in range(data.shape[1]-1): #this loop is only necessary when there is more than one feature
            for j in range(data.shape[0]):
                #make temp values
                # print("j is: ", j)
                # print("data is:", data[j,i])
                left_temp, right_temp = self.one_step_split(i, data[j,i], data)
                temp_mse = self.mean_squared_error(left_temp, right_temp)

                #check if either the left temp or right temp have a length of 0, if so move on to next iteration
                #print("came back to the get_best_split function")
                if (len(left_temp) == 0 or len(right_temp) == 0):
                    #print("there was 0 in left or right split")
                    continue
                
                # print("---------------------------")
                # print("testing mse: ", temp_mse)
                # print("best mse: ", best_mse)
                #check if this index yields a better mse and replace if so
                if (temp_mse < best_mse):
                    best_mse = temp_mse
                    feature_index = i
                    split_index = j
                    left_best = left_temp
                    right_best = right_temp
            

        #make the left and right node of the node that will be returned. These will be used in the next iteration of split
        left_node = Node(split_val=0, data=left_best, left=None, right=None)
        right_node = Node(split_val=0, data=right_best, left=None, right=None)

        # print("left", left_best)
        # print("right", right_best)

        node = Node(data[split_index, feature_index], data, left_node, right_node)

        return node



    @typechecked
    def one_step_split(
        self, index: int, value: float, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset based on an attribute and an attribute value
        index is the variable to be split on (left split < threshold)
        returns the left and right split each as list
        each list has elements as `rows' of the df
        """
        left_side = np.empty((0, data.shape[1]))
        right_side = np.empty((0, data.shape[1]))
        
        # print("-------------------------")
        # print("the value is:", value)
        # #go through the data and append based on if the data at that index is greater that or equal to or less than the given value
        for i in range(data.shape[0]):
            # print(data[i,index])
            # print(data[i])
            if(data[i,index] >= value):
                right_side = np.append(right_side, [data[i]], axis=0)
                # print("went right")
            else:
                left_side = np.append(left_side, [data[i]], axis=0)
                # print("went left")
            
        
        # print("left",left_side)
        # print("right",right_side)

        return left_side, right_side

@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:
    """
    Return True if node's value > row's value (of the variable)
    Else False
    """
    if (node.split_val > row[row.shape[0]-1]):
        return True
    else:
        return False



@typechecked
def predict(
    node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]
) -> float:
    #node represents the root node of the tree, row represents the (x,y) value from the data and the comparator calls the compare function

    #if there is no left and right nodes then 
    if (node.left == None and node.right == None):
        #need to return the average of the values in the node
        return np.mean(node.data[:,1])
    
    elif (comparator(node, row)):
        predict(node.right, row, comparator)

    else:
        predict(node.left, row, comparator)

    


class TreeClassifier(TreeRegressor):
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        root_node = TreeRegressor(self.data, self.max_depth)

        return root_node.build_tree()


    @typechecked
    def gini_index(
        self,
        left_split: np.ndarray,
        right_split: np.ndarray,
        classes: List[float],
    ) -> float:
        """
        Calculate the Gini index for a split dataset
        Similar to MSE but Gini index instead
        """
        #need to do a summation of proportion(proportion-1) for each proportion on the left and right side then add them up
        total_vals = len(left_split) + len(right_split)

        sum = np.sum(np.abs(1-left_split[:-1])) + np.sum(np.abs(1-right_split[:-1]))

        return sum/total_vals


    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset
        """
        classes = list(set(row[-1] for row in data))

        #doing the first split to get a base score to compare as I loop
        left_best = np.empty((1, data.shape[1]))
        right_best = np.empty((1, data.shape[1]))
        split_index = 0
        feature_index = 0
        best_gini = 1000.0

        temp_gini = 0.0


        for i in range(data.shape[1]-1): #this loop is only necessary when there is more than one feature
            for j in range(data.shape[0]):
                #make temp values
                # print("j is: ", j)
                # print("data is:", data[j,i])
                left_temp, right_temp = self.one_step_split(i, data[j,i], data)
                temp_gini = self.gini_index(left_temp, right_temp, classes)
                

                #check if either the left temp or right temp have a length of 0, if so move on to next iteration
                #print("came back to the get_best_split function")
                if (len(left_temp) == 0 or len(right_temp) == 0):
                    #print("there was 0 in left or right split")
                    continue
                
                # print("---------------------------")
                # print("testing mse: ", temp_mse)
                # print("best mse: ", best_mse)
                #check if this index yields a better mse and replace if so
                if (temp_gini < best_gini):
                    best_gini = temp_gini
                    feature_index = i
                    split_index = j
                    left_best = left_temp
                    right_best = right_temp
            

        #make the left and right node of the node that will be returned. These will be used in the next iteration of split
        left_node = Node(split_val=0, data=left_best, left=None, right=None)
        right_node = Node(split_val=0, data=right_best, left=None, right=None)

        # print("left", left_best)
        # print("right", right_best)

        node = Node(data[split_index, feature_index], data, left_node, right_node)

        return node


if __name__ == "__main__":
    # Question 1
    # filename = "Hitters.csv"  # Provide the path of the dataset
    # df = read_data(filename)
    # lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # max_iter = 1e8
    # final_features, final_label = data_preprocess(df)
    # x_train, x_test, y_train, y_test = data_split(
    #     final_features, final_label, 0.2
    # )
    # ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    # lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    # model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    # model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    
    # ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)
    # print("The area under curve measurement for Ridge Regression is", ridge_auc)

    # # # Plot the ROC curve of the Ridge Model. Include axes labels,
    # # # legend and title in the Plot. Any of the missing
    # # # items in plot will result in loss of points.

    #mostly ripped off from hw2
    #get tpr and fpr using roc_score function
    # ridge_reg_pred = model_R.predict(x_test)
    # ridge_reg_fpr, ridge_reg_tpr, ridge_threshold = roc_curve(y_test, ridge_reg_pred)

    # plt.plot(ridge_reg_fpr, ridge_reg_tpr, label='Ridge')
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("Ridge Regression ROC Curve")
    # plt.legend(loc='lower right')
    # plt.show()

    # lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)
    # print("The area under curve measurement for Lasso Regression is", lasso_auc)

    # # # Plot the ROC curve of the Lasso Model.
    # # # Include axes labels, legend and title in the Plot.
    # # # Any of the missing items in plot will result in loss of points.

    # #mostly ripped off from hw2
    # #get tpr and fpr using roc_score function
    # lasso_reg_pred = model_L.predict(x_test)
    # lasso_reg_fpr, lasso_reg_tpr, lasso_threshold = roc_curve(y_test, lasso_reg_pred)

    # plt.plot(lasso_reg_fpr, lasso_reg_tpr, label='Lasso')
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("Lasso Regression ROC Curve")
    # plt.legend(loc='lower right')
    # plt.show()

    # # SUB Q1
    data_regress = np.loadtxt("noisy_sin_subsample_2.csv", delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.show()

    #testing stuff ********************************************
    #print(data_regress.shape[0])

    print(data_regress)
    regressor = TreeRegressor(data_regress, 4)
    tree = regressor.build_tree()

    regressor.treeChecker(regressor.root_node, 1)

    # mse = 0.0
    # for data_point in data_regress:
    #     mse += (data_point[1] - predict(tree, data_point, compare_node_with_threshold)) ** 2

    # plt.figure()
    # plt.plot()
    # plt.xlabel("Depth")
    # plt.ylabel("MSE")
    # plt.show()
    #testing stuff ********************************************


    # mse_depths = []
    # for depth in range(1, 5): #DONT FORGET TO UNCOMMENT =
    #     regressor = TreeRegressor(data_regress, depth)
    #     tree = regressor.build_tree()
    #     mse = 0.0
    #     for data_point in data_regress:
    #         mse += (data_point[1] - predict(tree, data_point, compare_node_with_threshold)) ** 2
    #     mse_depths.append(mse / len(data_regress))
    # plt.figure()
    # plt.plot(mse_depths)
    # plt.xlabel("Depth")
    # plt.ylabel("MSE")
    # plt.show()

    # # SUB Q2
    # csvname = "new_circle_data.csv"  # Place the CSV file in the same directory as this notebook
    # data_class = np.loadtxt(csvname, delimiter=",")
    # data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    # plt.figure()
    # plt.scatter(
    #     data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    # )
    # plt.xlabel("Features, x1")
    # plt.ylabel("Features, x2")
    # plt.show()

    # accuracy_depths = []
    # for depth in range(1, 8):
    #     classifier = TreeClassifier(data_class, depth)
    #     tree = classifier.build_tree()
    #     correct = 0.0
    #     for data_point in data_class:
    #         correct += float(
    #             data_point[2]
    #             == predict(tree, data_point, compare_node_with_threshold)
    #         )
    #     accuracy_depths.append(correct / len(data_class))
    # # Plot the MSE
    # plt.figure()
    # plt.plot(accuracy_depths)
    # plt.xlabel("Depth")
    # plt.ylabel("Accuracy")
    # plt.show()
