#!/usr/bin/env python
# coding: utf-8

# ## Student Name: Camden L Hill
# ## Date: March 9, 2026
# 

# ## Package Imports
# 
# Import **all** required packages here. Do not add new imports in later cells.

# In[1]:


# ── Standard library ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

# ── Numerical / data ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualization ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── Machine learning ────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, adjusted_rand_score,
)

# ── Settings ────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
# get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)

# ── Extra Libraries ─────────────────────────────────────────────────────────
from os import getcwd, path
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,max_error
from sklearn.inspection import DecisionBoundaryDisplay
from itertools import combinations
from sklearn.utils import shuffle
# from sympy import *

print('All packages imported successfully.')

# ## Part 1 – Binary Classification with Logistic Regression

# ### 1a — Data Exploration

# In[50]:


# Load the sensor binary dataset
#  load data/p1a_sensor_binary.csv into a DataFrame called df_p1
# file_path_rel = "HW3 data\\p1a_sensor_binary.csv"
# current_dir = getcwd()
# full_path = "C" + str(path.join(current_dir, file_path_rel))[1:]
# print(full_path)
# print(path_str)
# for i in range(len(path_str)):
#     if path_str[i] != full_path[i]:
#         print(i,path_str[i],full_path[i])
#         full_path = full_path[:i] + "_" + full_path[i+1:]
#         break
# print(full_path == path_str)
print("There seems to be an issue with the Python OS standard library version introduced this February,\nso if you're grading this please just copy your path into the path_str.")
# df_p1 = pd.read_csv(full_path)

# current_dir = getcwd()
# file_path_rel = "drag_data.csv"
# full_path = path.join(current_dir, file_path_rel)
# df_p1 = pd.read_csv(full_path)

directory_str = r"C:\Users\hilla\OneDrive\Documents\College\YS52\AERO 689\HW\HW SolM\HW3 SolM\HW3_data"
file_str_p1 = "p1a_sensor_binary.csv"
df_p1 = pd.read_csv("{}\\{}".format(directory_str,file_str_p1))

# Display first 10 rows
#  display first 10 rows
df_p1.head(10)


# In[51]:


# Summary statistics grouped by label
#  compute and print summary statistics for each feature, grouped by label
df_p1_summary_stats = df_p1.describe()
numr_df_p1 = df_p1["label"].value_counts()
c_names_df_p1 = df_p1.columns
print(df_p1_summary_stats)


# In[52]:


# Scatter plot colored by class
#  scatter plot with vibration_rms on x-axis, exhaust_temp_deviation on y-axis
#       color class 0 (normal) and class 1 (degraded) differently
#       add title, axis labels, and legend
normal_df_p1 = df_p1[df_p1[c_names_df_p1[2]] == 0].copy()
degraded_df_p1 = df_p1[df_p1[c_names_df_p1[2]] == 1].copy()
plt.scatter(normal_df_p1[c_names_df_p1[0]],normal_df_p1[c_names_df_p1[1]],label="Class 0 (Normal)",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
plt.scatter(degraded_df_p1[c_names_df_p1[0]],degraded_df_p1[c_names_df_p1[1]],label="Class 1 (Degraded)",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
plt.xlabel("Vibration RMS")
plt.ylabel("Exhaust Temperature Deviation")
plt.title("Exhaust Temperature Deviation vs. Vibration RMS")
plt.grid()
plt.legend()
plt.show()


# **1a Analysis:**
# 
# [Your answer here: Are the classes visually separable? Do the feature distributions differ? What does this suggest about logistic regression suitability?]
# 
# The classes are obviously visually separable as the normal class 0 data is in the bottom left and the degraded class 1 data is in the top right with a lot of empty space in the middle of the graph. The distribution of the degraded data is spread out over a wider range of the x-axis compared to the normal data. The degraded data is also clustered over a larger range of the y-axis than the normal data, albeit to a lesser degree than the variation across the x-axis. The clear separation of the data and the complete lack of overlap means that logistic regression would an extremely suitable model to use to classify the data. 

# ### 1b — Train and Evaluate

# In[72]:


# Train/test split
#  split df_p1 into 80/20 train/test (random_state=42)
class ModelML:
    def __init__(self,model_type,features,y=None,random_state = None):
        self.model_type = model_type
        self.features = features
        self.feature_names = features.columns
        self.num_rows = len(self.features[self.feature_names[0]])
        self.y = y
        if isinstance(y,pd.Series):
            # print(len(y.shape))
            shape_y = y.shape
            if len(shape_y) > 1:
                self.y_names = y.columns
            # self.num_rows = shape_y[0]
        self.indices = range(self.num_rows)
        if random_state == None:
            self.random_state = np.random.default_rng(0,10**6)
        else:
            self.random_state = random_state
    
    def change_model_type(self,new_model_type):
        self.model_type = new_model_type

    def poly_X(self,degree,include_bias = False):
        self.poly = PolynomialFeatures(degree, include_bias)
        self.X = self.poly.fit_transform(self.features)
        self.d = self.X.shape[1]

    def custom_X(self,X):
        self.X = X
        # print(type(X))
        self.d = X.shape[1]

    def tt_split(self, test_size = 0.2): #, indices = None):
        self.X_train, self.X_test, self.y_train, self.y_test, self.train_indices, self.test_indices = \
        train_test_split(self.X, self.y, self.indices, test_size=test_size, random_state=self.random_state)
        # if indices == None:
        #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)
        # else:
        #     self.X_train, self.X_test, self.y_train, self.y_test, self.train_indices, self.test_indices = train_test_split(self.X, self.y, indices, test_size=test_size, random_state=self.random_state)
    
    # def kfold_model(self, n_splits = 10,test_size = 0.2,scale = "yes",kwargs_model={}):
    #     kf = KFold(n_splits,shuffle=True)
    #     kf.get_n_splits()
    #     kintercepts,kcoefficients = [[],[]]
    #     for i, (train_index, test_index) in enumerate(kf.split(self.X)):
    #         self.X_train = self.features.iloc[train_index]
    #         self.X_test = self.features.iloc[test_index]
    #         self.y_train = self.y.iloc[train_index]
    #         self.y_test = self.y.iloc[test_index]
    #         self.full_model_process(test_size,scale,kwargs_model)
    #         kintercepts.append(self.intercept)
    #         kcoefficients.append(self.coefficients)
    #     self.intercept = np.mean(kintercepts)
    #     self.coefficients = np.mean(np.array(kcoefficients),axis=0)

    def create_model(self,kwargs_model={}):
        if self.model_type == "LinearRegression":
            self.model= LinearRegression().fit(self.X_train_main,self.y_train_main,**kwargs_model)
        elif self.model_type == "LogisticRegression":
            self.model = LogisticRegression(random_state=self.random_state,**kwargs_model).fit(self.X_train_main,self.y_train_main)
        elif self.model_type == "K-Clustering":
            self.model = KMeans(random_state=self.random_state,**kwargs_model).fit(self.X_main)
    
    def set_main_model_Xy(self,scale="yes"):
        if self.model_type in ["LinearRegression","LogisticRegression"]: # Supervised Models
            self.y_train_main = self.y_train
            self.y_test_main = self.y_test
            if scale == "yes":
                self.X_test_main = self.X_test_scaled
                self.X_train_main = self.X_train_scaled
            elif scale == "no":
                self.X_test_main = self.X_test
                self.X_train_main = self.X_train
        elif self.model_type in ["K-Clustering"]: # Unsupervised Models
            if scale == "yes":
                self.X_main = self.X_scaled
            elif scale == "no":
                self.X_main = self.X

    def predict_model(self):
        if self.model_type in ["LinearRegression","LogisticRegression"]: # Supervised Models
            self.model_prediction = pd.DataFrame(self.model.predict(self.X_test_main),index=self.test_indices)
        elif self.model_type in ["K-Clustering"]: # Unsupervised Models
            self.model_prediction = pd.DataFrame(self.model.predict(self.X_main))
        # if self.model_type == "LogisticRegression":
        #     self.predict_proba(self.X_test)
    #######################################################################################################################
    def sd_scale(self):
        self.scaler = StandardScaler()
        if self.model_type in ["LinearRegression","LogisticRegression"]: # Supervised Models
            self.X_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.X_train),columns=self.feature_names,index=self.train_indices)
            self.X_test_scaled = pd.DataFrame(self.scaler.transform(self.X_test),columns=self.feature_names,index=self.test_indices)
            # self.X_scaled = self.scaler.transform(self.X)
        elif self.model_type in ["K-Clustering"]: # Unsupervised Models
            self.X_scaled = pd.DataFrame(self.scaler.fit_transform(self.X),columns=self.feature_names)
        self.mean = self.scaler.mean_
        self.SD = self.scaler.scale_

    def data_to_PCA(self,kwargs_PCA={}):
        self.pca = PCA(random_state=self.random_state,**kwargs_PCA)
        if self.model_type in ["LinearRegression","LogisticRegression"]: # Supervised Models
            self.X_train_scaled = pd.DataFrame(self.pca.fit_transform(self.X_train_scaled),index=self.train_indices)
            self.X_test_scaled = pd.DataFrame(self.pca.transform(self.X_test_scaled),index=self.test_indices)
            # self.X_scaled = self.scaler.transform(self.X)
        elif self.model_type in ["K-Clustering"]: # Unsupervised Models
            self.X_scaled = pd.DataFrame(self.pca.fit_transform(self.X_scaled))
            # print(self.X_scaled)
        self.mean = self.scaler.mean_
        self.SD = self.scaler.scale_
        self.feature_names = self.X_scaled.columns # ["{}".format(i) for i in range(kwargs_PCA["n_components"])]
    #######################################################################################################################
    def initialize_model(self, test_size=0.2, scale="yes",pca="no",kwargs_model={},kwargs_pca={}):
        self.tt_split(test_size)
        if scale == "yes":
            self.sd_scale()
        if pca == "yes":
            self.data_to_PCA(kwargs_pca)
        self.set_main_model_Xy(scale)
        self.create_model(kwargs_model)

    def full_model_process(self,test_size = 0.2,scale = "yes",pca="no",kwargs_model={},kwargs_pca={}):
        self.initialize_model(test_size,scale,pca,kwargs_model,kwargs_pca)
        self.predict_model()
        self.coefficients_model()
        if pca == "no":
            if scale == "yes":
                self.cnvrt_to_real_coefficient_model()

    def generate_results_model(self):
        self.model_results = (self.X @ self.true_coefficients) + self.true_intercept

    def coefficients_model(self,biased = "no"):
        if biased == "no":
            self.intercept = self.model.intercept_
            self.coefficients = self.model.coef_
        else:
            self.intercept = self.model.coef_[:,0]
            self.coefficients = self.model.coef_[:,1:]
            # print(self.intercept)
            # print(self.coefficients)

    def cnvrt_to_real_coefficient_model(self):
        # if biased == "no":
        self.true_coefficients =  self.coefficients / self.SD # self.scaler.inverse_transform
        self.true_intercept = self.intercept - np.sum((self.coefficients * self.mean) / self.SD)
        # else:
        #     self.true_coefficients =  self.coefficients[1:] / self.SD # self.scaler.inverse_transform
        #     self.true_intercept = self.coefficients[0] - np.sum((self.coefficients[1:] * self.mean) / self.SD)
    #######################################################################################################################
    def initialize_model_clustering(self, scale="yes",pca="no",kwargs_model={},kwargs_pca={}):
        self.custom_X(self.features)
        if scale == "yes":
            self.sd_scale()
        if pca == "yes":
            self.data_to_PCA(kwargs_pca)
        self.set_main_model_Xy(scale)
        self.create_model(kwargs_model)

    def full_model_process_clustering(self,scale = "yes",pca="no",kwargs_model={},kwargs_pca={}):
        self.initialize_model_clustering(scale,pca,kwargs_model,kwargs_pca)
        self.predict_model()
        self.labels = pd.DataFrame(self.model.labels_)
        self.WCSS = self.model.inertia_
        if pca == "no":
            if scale == "yes":
                self.cnvrt_clusters_ifscaled()
            else:
                self.cnvrt_clusters_ifunscaled()
        else:
            if scale == "yes":
                self.cnvrt_clusters_ifunscaled()
        # print(self.X_main)

    def cnvrt_clusters_ifunscaled(self):
        self.cluster_centers_unscaled = self.model.cluster_centers_

    def cnvrt_clusters_ifscaled(self):
        self.cluster_centers_unscaled = self.scaler.inverse_transform(self.model.cluster_centers_) # (self.model.cluster_centers_ * self.SD) + self.mean
        self.cluster_centers_scaled = self.model.cluster_centers_
    #######################################################################################################################
    def metrics_lin_reg(self):
        n_test = len(self.y_test_main)
        self.mse = mean_squared_error(self.y_test_main, self.model_prediction)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_test_main, self.model_prediction)
        self.r2a = 1 - ((1-self.r2)*(n_test-1)/(n_test - (self.d + 1) - 1)) # plus 1 for d to account for intercept
        self.mae = mean_absolute_error(self.y_test, self.model_prediction)
        self.mape = mean_absolute_percentage_error(self.y_test_main, self.model_prediction)
        self.maxe = max_error(self.y_test_main, self.model_prediction)
    
    def print_lin_reg(self):
        print("Key Metrics: MSE,                  RMSE,                  R^2,                R_adj^2,            MAE,                   MAPE,                 Max Error")
        print("             {}, {}, {}, {}, {}, {}, {}".format(self.mse,self.rmse,self.r2,self.r2a,self.mae,self.mape,self.maxe))
    
    def confusion_matrix_calculation(self): # ,col_names=None):
        if self.model_type in ["LinearRegression","LogisticRegression"]: # Supervised Models
            self.confusion_matrix = confusion_matrix(self.y_test_main,self.model_prediction)
        elif self.model_type in ["K-Clustering"]: # Unsupervised Models
            self.confusion_matrix = confusion_matrix(self.y,self.labels) # pd.DataFrame(confusion_matrix(self.y_test,self.model_prediction)) #,columns = col_names)

    def key_metrics_classifiers(self): # ,col_names=None):
        self.confusion_matrix_calculation() # ,col_names)
        self.accuracy = accuracy_score(self.y_test_main,self.model_prediction)
        self.precision = precision_score(self.y_test_main,self.model_prediction,average="macro")
        self.recall = recall_score(self.y_test_main,self.model_prediction,average="macro")
        self.specificity = recall_score(self.y_test_main,self.model_prediction,average="macro", pos_label=0)
        self.F1_score = f1_score(self.y_test_main,self.model_prediction,average="macro")
        # TN,FP,FN,TP = [self.confusion_matrix[0,0],self.confusion_matrix[0,1],self.confusion_matrix[1,0],self.confusion_matrix[1,1]]
        # print(TN,FP,FN,TP)
        # self.accuracy = (TP+TN) / (TP+TN+FP+FN)
        # self.precision = TP / (TP+FP)
        # self.recall = TP / (TP+FN)
        # self.specificity = TN / (TN + FP)
        # self.F1_score = 2*(self.precision * self.recall) / (self.precision + self.recall)

    def print_key_metrics_classifers(self):
        print("Accuracy = {}".format(self.accuracy))
        print("Precision = {}".format(self.precision))
        print("Recall = {}".format(self.recall))
        print("Specificity = {}".format(self.specificity))
        print("F1-Score = {}".format(self.F1_score))

    def key_metrics_clustering(self,labels_cols = 0):
        if labels_cols == 0:
            self.silhouette_score = silhouette_score(self.X_main,self.labels[0],random_state=self.random_state)
        else:
            self.silhouette_score = silhouette_score(self.X_main,self.labels[labels_cols],random_state=self.random_state)
        if isinstance(self.y,pd.Series):
            if labels_cols == 0:
                self.ARI = adjusted_rand_score(self.y,self.labels[0])
            else:
                self.ARI = [adjusted_rand_score(self.y[i],self.labels[i]) for i in range(labels_cols)]

    def print_key_metrics_clustering(self):
        print("Silhouette Score = {}".format(self.silhouette_score))
        if isinstance(self.y,pd.Series):
            print("Adjusted Random Score (ARI) = {}".format(self.ARI))
    #######################################################################################################################
    def sigmoid(self,z):
        """Element-wise sigmoid function."""
        return 1/(1+np.exp(-z))

    def binary_cross_entropy_log_reg(self,yvals,sig_vals): # ,y_series=None):
        # if not hasattr(self,"sig_vals"):
        #     z = self.X_train_main @ self.beta
        #     self.sigmoid(z)
        self.Jbeta = (-1/len(yvals)) * np.sum((yvals*np.log(sig_vals)) + ((1-yvals)*np.log(1-sig_vals))) # self.Jbeta = (-1/self.num_rows)*(self.y*np.log(self.sigmoid))

    def gradient_log_reg(self,x_vals,y_vals,sig_vals): # ,y_series=None):
        # if not hasattr(self,"sig_vals"):
        #     z = self.X_train_main @ self.beta
        #     self.sigmoid(z)
        return (1/len(y_vals)) * (x_vals.T @ (sig_vals - y_vals))
        
    def gradient_descent_log_reg(self,model="SGD",learning_rate=0.1,iterations = 1000,scale = "yes"):
        self.tt_split()
        if scale == "yes":
            self.sd_scale()
        self.set_main_model_Xy(scale)
        self.X_train_main.insert(0,"Bias",np.ones(len(self.X_train_main[self.feature_names[0]])))
        self.X_test_main.insert(0,"Bias",np.ones(len(self.X_test_main[self.feature_names[0]])))

        self.beta = np.zeros(self.X_train_main.shape[1])
        self.GD_betas,self.GD_cost = [[],[]]
        if model == "GD":
            for i in range(0,iterations):
                z = self.X_train_main @ self.beta
                sig_vals = self.sigmoid(z)
                gradient = self.gradient_log_reg(self.X_train_main,self.y_train_main,sig_vals)
                self.beta = np.array(self.beta - (learning_rate*gradient)) # -= (learning_rate*self.grad) # np.array(self.beta - (learning_rate*self.grad))
                sig_vals = self.sigmoid(self.X_train_main @ self.beta)
                self.binary_cross_entropy_log_reg(self.y_train_main,sig_vals)
                self.GD_betas.append(self.beta.copy())
                self.GD_cost.append(self.Jbeta)
        elif model == "SGD":
            rows_training = len(self.train_indices)
            inv_rows = 1/rows_training
            for i in range(0,iterations):
                temp_X_train_main,temp_y_train_main = shuffle(self.X_train_main,self.y_train_main,random_state = self.random_state+i)
                for j in range(rows_training):
                    x_vals,y_vals = [temp_X_train_main.iloc[j].values,temp_y_train_main.iloc[j]]
                    z = x_vals @ self.beta
                    sig_val = self.sigmoid(z)
                    gradient = (x_vals * (sig_val - y_vals))*inv_rows
                    self.beta = np.array(self.beta - (learning_rate*gradient))
                self.binary_cross_entropy_log_reg(temp_y_train_main,self.sigmoid(temp_X_train_main @ self.beta))
                self.GD_betas.append(self.beta.copy())
                self.GD_cost.append(self.Jbeta)
                
        
        self.intercept = self.GD_betas[iterations-1][0]
        self.coefficients = self.GD_betas[iterations-1][1:]
        if scale == "yes":
            self.cnvrt_to_real_coefficient_model()
    #######################################################################################################################    
    def decision_boundary_values(self,x_indice,y_indice,x_vars_variable = [],x_vars_eqs = ["x","x**2"],x_vars_fixed = [],scale = "no",num_x_vals=300):
        if scale == "no":
            reg_coeffs,reg_intercept,X_dataframe = [self.true_coefficients,self.true_intercept,self.X_train]
        else:
            reg_coeffs,reg_intercept,X_dataframe = [self.coefficients,self.intercept,self.X]
        fixed_excluded_list = x_vars_variable+[x_indice,y_indice]
        self.num_classes = reg_coeffs.shape[0]
        num_features = X_dataframe.shape[1]
        means = X_dataframe.mean().values # [np.mean(X_dataframe) for i in range(num_features)]
        for i in range(len(x_vars_fixed)):
            if i not in fixed_excluded_list:
                means[i] = x_vars_fixed[i]
        x_vals = np.linspace(X_dataframe.iloc[:,x_indice].min(),X_dataframe.iloc[:,x_indice].max(),num_x_vals) # X_dataframe[self.feature_names[x_indice]] # 
        yvals = [] # ,ind = [[],0]
        if self.num_classes == 1:
            reg_coeffs = reg_coeffs[0]
            val = reg_intercept[0]
            for k in range(num_features):
                if k not in fixed_excluded_list:
                    val += reg_coeffs[k] * means[k]
            if x_vars_variable == []:
                yvals.append(-(val + reg_coeffs[x_indice]*x_vals)/reg_coeffs[y_indice])
            # else:
            #     non_linear_x_vals = reg_coeffs[x_vars_variable[0]] * x_vals
            #     if len(x_vars_variable) > 1:
            #         for i in x_vars_variable[1:]:
            #             non_linear_x_vals += reg_coeffs[i] * X_dataframe[self.feature_names[i]]
            #     yvals.append(-(val + ((reg_coeffs[x_indice]*x_vals) + non_linear_x_vals))/reg_coeffs[y_indice])
        else:
            for i,j in combinations(range(self.num_classes),2): # has to be pariwise so using combinations function
                pair_coeff = reg_coeffs[i] - reg_coeffs[j]
                pair_inter = reg_intercept[i] - reg_intercept[j]
                val = pair_inter
                for k in range(num_features):
                    if k not in [x_indice, y_indice]:
                        val += pair_coeff[k] * means[k]
                # print(-(val + pair_coeff[x_indice]*x_vals)/pair_coeff[y_indice])
                yvals.append(-(val + pair_coeff[x_indice]*x_vals)/pair_coeff[y_indice])

        self.decision_xvals = x_vals # X_dataframe[self.feature_names[x_indice]]
        self.decision_yvals = yvals

    def grid_decision_boundary_values(self,x_indice,y_indice,scale = "no",num_x_vals=300):
        if scale == "no":
            reg_coeffs,reg_intercept,X_dataframe = [self.true_coefficients,self.true_intercept,self.X_train]
        else:
            reg_coeffs,reg_intercept,X_dataframe = [self.coefficients,self.intercept,self.X]
        # fixed_excluded_list = x_vars_variable+[x_indice,y_indice]
        self.num_classes = 1 # reg_coeffs.shape[0]
        # num_features = X_dataframe.shape[1]
        means = X_dataframe.mean().values # [np.mean(X_dataframe) for i in range(num_features)]
        # for i in range(len(x_vars_fixed)):
        #     if i not in fixed_excluded_list:
        #         means[i] = x_vars_fixed[i]
        x_vals = np.linspace(X_dataframe.iloc[:,x_indice].min(),X_dataframe.iloc[:,x_indice].max(),num_x_vals)
        y_vals = np.linspace(X_dataframe.iloc[:,y_indice].min(),X_dataframe.iloc[:,y_indice].max(),num_x_vals) # X_dataframe[self.feature_names[x_indice]] # np.linspace(X_dataframe.iloc[:,x_indice].min(),X_dataframe.iloc[:,x_indice].max(),num_x_vals)
        xx_vals, yy_vals = np.meshgrid(x_vals, y_vals)
        Zmat = np.zeros(xx_vals.shape)
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                features = means.copy()
                features[x_indice] = xx_vals[i,j]
                features[y_indice] = yy_vals[i,j]
                features[2] = xx_vals[i,j]**2
                features[3] = yy_vals[i,j]**2
                Zmat[i,j] = 1/(1+ np.exp(-(reg_intercept + np.dot(features, reg_coeffs.flatten()))))# reg_intercept + np.dot(reg_coeffs, features)
        self.decision_xvals = xx_vals
        self.decision_yvals = yy_vals
        # print(xx_vals)
        # print(yy_vals)
        self.decision_surface = Zmat
        # if self.num_classes == 1:
        #     reg_coeffs = reg_coeffs[0]
        #     for k in range(num_features):
        #         if k not in fixed_excluded_list:
        #             c_val += reg_coeffs[k] * means[k]
        #     if x_vars_variable == []:
        #         yvals.append(-(val + reg_coeffs[x_indice]*x_vals)/reg_coeffs[y_indice])
        #     else:
        #         non_linear_x_vals = reg_coeffs[x_vars_variable[0]] * X_dataframe[self.feature_names[x_vars_variable[0]]]
        #         if len(x_vars_variable) > 1:
        #             for i in x_vars_variable[1:]:
        #                 non_linear_x_vals += reg_coeffs[i] * X_dataframe[self.feature_names[i]]
        #         yvals.append(-(val + ((reg_coeffs[x_indice]*x_vals) + non_linear_x_vals))/reg_coeffs[y_indice])
        # else:
        #     for i,j in combinations(range(self.num_classes),2): # has to be pariwise so using combinations function
        #         pair_coeff = reg_coeffs[i] - reg_coeffs[j]
        #         pair_inter = reg_intercept[i] - reg_intercept[j]
        #         val = pair_inter
        #         for k in range(num_features):
        #             if k not in [x_indice, y_indice]:
        #                 val += pair_coeff[k] * means[k]
        #         # print(-(val + pair_coeff[x_indice]*x_vals)/pair_coeff[y_indice])
        #         yvals.append(-(val + pair_coeff[x_indice]*x_vals)/pair_coeff[y_indice])
    #######################################################################################################################
    def calculate_label_bmask_clustering(self,y_series):
        # print(y_series)
        if isinstance(y_series,pd.DataFrame):
            y_series = pd.Series(y_series[0])
        self.label_vals = np.sort(y_series.unique())
        self.label_nums = len(self.label_vals)
        self.label_bmasks = [y_series == self.label_vals[i] for i in range(self.label_nums)]

    def plot_labels_show(self,xlabel_plt,ylabel_plt,title_plt):
        plt.xlabel(xlabel_plt)
        plt.ylabel(ylabel_plt)
        plt.title(title_plt)

        # plt.grid()
        plt.show()

    def plot_labels_show_sub(self,obj,xlabel_plt,ylabel_plt,title_plt): #,legend_plt="yes"):
        obj.supxlabel(xlabel_plt)
        obj.supylabel(ylabel_plt)
        obj.suptitle(title_plt)
        # if legend_plt == "yes":
        #     plt.legend()
        # plt.grid()
        plt.show()

    def plot_clustering(self,obj,X_dataframe,x_indice,y_indice,label_colors = [],s=80,label_labels = [],legend_plt="yes"):
        for i in range(len(label_labels),self.label_nums):
            label_labels.append("Label {}".format(i+1))
        for i in range(len(label_colors),self.label_nums):
            label_colors.append((np.random.random(), np.random.random(), np.random.random()))
        for i in range(self.label_nums):
            # print(len(X_dataframe[self.feature_names[x_indice]]))
            # print(len(self.label_bmasks[i]))
            obj.scatter(X_dataframe[self.feature_names[x_indice]][self.label_bmasks[i]],X_dataframe[self.feature_names[y_indice]][self.label_bmasks[i]],s=s,label=label_labels[i],color=label_colors[i])
        if legend_plt == "yes":
            obj.legend()

    def plot_clusters_same(self,obj,x_indice,y_indice,scale="no",label=None,s=200,marker="*",color="black",edgecolors="white"):
        if scale == "no":
            obj.scatter(self.cluster_centers_unscaled[:,x_indice],self.cluster_centers_unscaled[:,y_indice],label=label,s=s,marker=marker,color=color,edgecolors=edgecolors)
            # for i in range(self.label_nums):
            #     plt.scatter(self.cluster_centers_unscaled[]
        elif scale == "yes":
            obj.scatter(self.cluster_centers_scaled[:,x_indice],self.cluster_centers_unscaled[:,y_indice],label=label,s=s,marker=marker,color=color,edgecolors=edgecolors)
        # if legend_plt == "yes":
        #     plt.legend()

    def plot_decision_boundary(self,boundary_indices=[],label="Decision Boundary",color=None):
        if boundary_indices == []:
            if self.num_classes == 1:
                boundary_indices = [0]
            else:
                boundary_indices = list(range(self.label_nums))
        if color == None:
            colors = [(np.random.random(), np.random.random(), np.random.random()) for i in boundary_indices]
        else:
            colors = ["black" for i in boundary_indices]
        # print(boundary_indices)
        # print(self.decision_yvals)
        for i in boundary_indices:
            plt.plot(self.decision_xvals,self.decision_yvals[i],label="{} {}".format(label,i+1),color=colors[i])

    def plot_decision_boundary_contour(self,boundary_indices=[],label="Decision Boundary",color=None):
        if boundary_indices == []:
            if self.num_classes == 1:
                boundary_indices = [0]
            else:
                boundary_indices = list(range(self.label_nums))
        if color == None:
            colors = [(np.random.random(), np.random.random(), np.random.random()) for i in boundary_indices]
        else:
            colors = ["black" for i in boundary_indices]
        # print(boundary_indices)
        # print(self.decision_yvals)
        for i in boundary_indices:
            plt.contour(self.decision_xvals,self.decision_yvals,self.decision_surface,label="{} {}".format(label,i+1),levels=[0.5],colors=colors)# label="{} {}".format(label,i+1),color=colors[i])
    #######################################################################################################################    


# In[73]:


X = df_p1[c_names_df_p1[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p1[c_names_df_p1[-1]]  # label column
p1a = ModelML("LogisticRegression",X,y,random_state = 42) # "LogisticRegression",

# X_train, X_test, y_train, y_test = None, None, None, None


# In[74]:


# Train logistic regression
#  train LogisticRegression(max_iter=1000, random_state=42)
# clf_p1b = None
p1a.custom_X(X)
p1a.full_model_process(kwargs_model = {"max_iter":1000})
# print(p1a.features)
print(p1a.X_train_main[p1a.feature_names][:4])
print(p1a.X_test_main[p1a.feature_names][:4])
print(p1a.y_train_main[:4])
print(p1a.y_test_main[:4])
print(p1a.true_intercept,p1a.true_coefficients)
# print(p1a.)

# Predict on test set
# y_pred_p1b = None


# In[75]:


# Confusion matrix
#  compute and display the confusion matrix
#       label rows/columns as ['Normal', 'Degraded']
# p1a.confusion_matrix_calculation(col_names=['Normal', 'Degraded']) # = pd.DataFrame(confusion_matrix(p1a.y_test,p1a.model_prediction),columns=['Normal', 'Degraded']) # ['Degraded', 'Normal'])
p1a.key_metrics_classifiers()

pd.DataFrame(p1a.confusion_matrix,index=["Actual Normal","Actual Degraded"],columns=['Predicted Normal', 'Predicted Degraded'])


# In[76]:


# Classification metrics
#  print accuracy, precision, recall, and F1-score for the degraded class
p1a.print_key_metrics_classifers()


# In[77]:


# Decision boundary plot
#  re-plot the training data scatter
#  overlay the decision boundary line
#       (solve for x2 as a function of x1 using model coefficients and intercept)
normal_df_p1 = df_p1[df_p1[c_names_df_p1[2]] == 0].copy()
degraded_df_p1 = df_p1[df_p1[c_names_df_p1[2]] == 1].copy()
# print(np.shape(np.array(p1a.X[p1a.feature_names[0]])))
print(p1a.true_intercept)
print(p1a.true_coefficients)
x2_vals = -(p1a.true_intercept[0] + (p1a.true_coefficients[0][0] * np.array(p1a.X[p1a.feature_names[0]]))) / p1a.true_coefficients[0][1]

# disp = DecisionBoundaryDisplay.from_estimator(
#     p1a.model, p1a.X, response_method="predict",
#     #xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
#     alpha=0.5)
# disp.ax_.scatter(p1a.X[p1a.feature_names[0]][:],p1a.X[p1a.feature_names[1]][:], edgecolor="k")

p1a.calculate_label_bmask_clustering(p1a.y)
p1a.plot_clustering(plt,p1a.X,0,1,label_colors=["red","blue","green","orange"],label_labels=["Class 0 (Normal)","Class 1 (Degraded)"])
# plt.scatter(normal_df_p1[c_names_df_p1[0]],normal_df_p1[c_names_df_p1[1]],label="Class 0 (Normal)",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(degraded_df_p1[c_names_df_p1[0]],degraded_df_p1[c_names_df_p1[1]],label="Class 1 (Degraded)",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
plt.plot(p1a.X[p1a.feature_names[0]],x2_vals,label="Decision Boundary",color="black")
p1a.plot_labels_show("Vibration RMS","Exhaust Temperature Deviation","Exhaust Temperature Deviation vs. Vibration RMS")
# plt.xlabel("Vibration RMS")
# plt.ylabel("Exhaust Temperature Deviation")
# plt.title("Exhaust Temperature Deviation vs. Vibration RMS")
# plt.grid()
# plt.legend()
# plt.show()


# **1b Analysis:**
# 
# **Coefficient interpretation:** [Your answer here: what do the signs of β₁ and β₂ tell you?]
# 
# **Threshold trade-off:** [Your answer here: false negatives vs false positives in a maintenance context; how to adjust the threshold?]
# 
# Both of the coefficients, once all the regression coefficients have been converted from their scaled form back to their normal form, are positive. This can be interpreted as that both model variables increase, for this specific model it is vibration RMS and exhaust temperature deviation, the bearings are more likely to be considered degraded. The negative intercept indicates that at low variables the probability of the bearing being normal is high. A model prediction of a class zero point would thus be have a sigmoid of less than 0.5.
# 
# There is a lot between the two clusters of data points, so there should almost never be a concern about predicting a false negative or false positive. However, to adjust the interpreted probability, the logistic regression class in Scikit has a function that predicts the probability of each point. You can then manually apply a cutoff where certain probabilities above or below a certain threshold beside 0.5 are used to change the labels. This allows for more user input into defining where the decision boundary lies.
# 

# ### 1c — Multiclass Classification: Flight Phases

# In[78]:


# Load flight phase data
#  load data/p1b_flight_phases.csv into df_p1c
file_str_p1b = "p1b_flight_phases.csv"
df_p1b = pd.read_csv("{}\\{}".format(directory_str,file_str_p1b))
print(df_p1b.head(10))

df_p1b_summary_stats = df_p1b.describe()
c_names_df_p1b = df_p1b.columns
numr_df_p1b = df_p1b[c_names_df_p1b[0]].value_counts()
print(df_p1b_summary_stats)

# Scatter plot colored by phase
#  scatter plot, altitude_rate vs airspeed_deviation, colored by phase label
phase0_df_p1b = df_p1b[df_p1b[c_names_df_p1b[2]] == 0].copy()
phase1_df_p1b = df_p1b[df_p1b[c_names_df_p1b[2]] == 1].copy()
phase2_df_p1b = df_p1b[df_p1b[c_names_df_p1b[2]] == 2].copy()
plt.scatter(phase0_df_p1b[c_names_df_p1b[0]],phase0_df_p1b[c_names_df_p1b[1]],label="Phase 0",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
plt.scatter(phase1_df_p1b[c_names_df_p1b[0]],phase1_df_p1b[c_names_df_p1b[1]],label="Phase 1",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
plt.scatter(phase2_df_p1b[c_names_df_p1b[0]],phase2_df_p1b[c_names_df_p1b[1]],label="Phase 2",color="green") # ,color=(np.random.random(), np.random.random(), np.random.random()))
plt.xlabel("Altitude Rate")
plt.ylabel("Airspeed Deviation")
plt.title("Altitude Rate vs. Airspeed Deviation")
plt.grid()
plt.legend()
plt.show()


# In[79]:


# Train one-vs-rest multiclass logistic regression
#  split 80/20, train LogisticRegression(max_iter=1000)
#       sklearn uses one-vs-rest by default for multiclass problems
X = df_p1b[c_names_df_p1b[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p1b[c_names_df_p1b[-1]]  # label column
p1b = ModelML("LogisticRegression",X,y,random_state = 42) # "LogisticRegression",

p1b.custom_X(X)
p1b.full_model_process(kwargs_model = {"max_iter":1000})
# clf_p1c = None
# y_pred_p1c = None
print(p1b.model.predict_proba(p1b.X_test))
print(p1b.model.score(p1b.X_test,p1b.y_test))


# In[80]:


# Confusion matrix (3x3)
#  display 3x3 confusion matrix, label rows/columns ['Descent','Cruise','Climb']
p1b.key_metrics_classifiers()
p1b.print_key_metrics_classifers()
pd.DataFrame(p1b.confusion_matrix,index=['Actual Descent','Actual Cruise','Actual Climb'],columns=['Predicted Descent','Predicted Cruise','Predicted Climb'])


# In[81]:


# Decision boundaries for all three phases
#  scatter plot with three decision boundary lines overlaid
#       use meshgrid + contourf, or plot analytical boundaries

print(p1b.true_intercept)
print(p1b.true_coefficients)

w1,w2,w3 = [p1b.true_coefficients[0]-p1b.true_coefficients[1],p1b.true_coefficients[1]-p1b.true_coefficients[2],p1b.true_coefficients[0]-p1b.true_coefficients[2]]
b1,b2,b3 = [p1b.true_intercept[0]-p1b.true_intercept[1],p1b.true_intercept[1]-p1b.true_intercept[2],p1b.true_intercept[0]-p1b.true_intercept[2]]

x2_vals1 = -(b1 + (w1[0] * np.array(p1b.X[p1b.feature_names[0]]))) / w1[1]
x2_vals2 = -(b2 + (w2[0] * np.array(p1b.X[p1b.feature_names[0]]))) / w2[1]
x2_vals3 = -(b3 + (w3[0] * np.array(p1b.X[p1b.feature_names[0]]))) / w3[1]

p1b.calculate_label_bmask_clustering(p1b.y)
p1b.plot_clustering(plt,p1b.X,0,1,label_colors=["red","blue","green","orange"],label_labels=["Phase 0","Phase 1","Phase 2"])
# plt.scatter(phase0_df_p1b[c_names_df_p1b[0]],phase0_df_p1b[c_names_df_p1b[1]],label="Phase 0",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(phase1_df_p1b[c_names_df_p1b[0]],phase1_df_p1b[c_names_df_p1b[1]],label="Phase 1",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(phase2_df_p1b[c_names_df_p1b[0]],phase2_df_p1b[c_names_df_p1b[1]],label="Phase 2",color="green") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.plot(p1b.X[p1b.feature_names[0]],x2_vals1,label="Decision Boundary 1",color="black")
# plt.plot(p1b.X[p1b.feature_names[0]],x2_vals2,label="Decision Boundary 2",color="brown")
# plt.plot(p1b.X[p1b.feature_names[0]],x2_vals3,label="Decision Boundary 3",color="orange")

p1b.decision_boundary_values(0,1)
p1b.plot_decision_boundary()
# plt.plot(p1b.X[p1b.feature_names[0]],x2_vals1,label="Decision Boundary 1",color="black")
# plt.plot(p1b.X[p1b.feature_names[0]],x2_vals2,label="Decision Boundary 2",color="brown")
# plt.plot(p1b.X[p1b.feature_names[0]],x2_vals3,label="Decision Boundary 3",color="orange")

p1b.plot_labels_show("Altitude Rate","Airspeed Deviation","Airspeed Deviation vs. Altitude Rate")
# plt.xlabel("Altitude Rate")
# plt.ylabel("Airspeed Deviation")
# plt.title("Altitude Rate vs. Airspeed Deviation")
# plt.grid()
# plt.legend()
# plt.show()


# **1c Analysis:**
# 
# [Your answer here: Why does one-vs-rest train three classifiers? What does each one learn? Which phases are most often confused?]
# 
# One-vs-rest trains three classifiers because there are three combinations of labels. Therefore, there is a decision boundary between Label 1 and Label 2, Label 2 and Label 3, and Label 1 and Label 3. For this specific problem, the phases (labels) that are most confused are Phase 0 and Phase 2. This is because some data points from each label overlap with the cluster of Phase 1 data points, so the model isn’t able to distinguish between the groups. 

# ---
# ## Part 2 – Effect of Regularization

# In[82]:


# Load regularization dataset
#  load data/p1c_regularization.csv into df_p2
file_str_p1c = "p1c_regularization.csv"
df_p1c = pd.read_csv("{}\\{}".format(directory_str,file_str_p1c))
print(df_p1c.head(10))

df_p1c_summary_stats = df_p1c.describe()
c_names_df_p1c = df_p1c.columns
numr_df_p1c = df_p1c[c_names_df_p1c[0]].value_counts()
print(df_p1c_summary_stats)

# Scatter plot — identify outlier region
#  scatter feature_1 vs feature_2, colored by label
label0_df_p1c = df_p1c[df_p1c[c_names_df_p1c[2]] == 0].copy()
label1_df_p1c = df_p1c[df_p1c[c_names_df_p1c[2]] == 1].copy()
plt.scatter(label0_df_p1c[c_names_df_p1c[0]],label0_df_p1c[c_names_df_p1c[1]],label="Label 0",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
plt.scatter(label1_df_p1c[c_names_df_p1c[0]],label1_df_p1c[c_names_df_p1c[1]],label="Label 1",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Feature 2 vs. Feature 1")
plt.grid()
plt.legend()
plt.show()


# In[83]:


# Train three models with different C values
#  split 80/20 (random_state=42)
#  train LogisticRegression for C = [0.01, 1.0, 100.0]
C_values = [0.01, 1.0, 100.0]
models_p2 = {}

X = df_p1c[c_names_df_p1c[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p1c[c_names_df_p1c[-1]]  # label column
p1c_C1 = ModelML("LogisticRegression",X,y,random_state = 42)
p1c_C2 = ModelML("LogisticRegression",X,y,random_state = 42)
p1c_C3 = ModelML("LogisticRegression",X,y,random_state = 42)

p1c_C1.custom_X(X)
p1c_C2.custom_X(X)
p1c_C3.custom_X(X)
p1c_C1.full_model_process(kwargs_model = {"max_iter":1000,"C":C_values[0]})
p1c_C2.full_model_process(kwargs_model = {"max_iter":1000,"C":C_values[1]})
p1c_C3.full_model_process(kwargs_model = {"max_iter":1000,"C":C_values[2]})

print(p1c_C1.model.predict_proba(p1c_C1.X_test)[:5,:])
print(p1c_C1.model.score(p1c_C1.X_test,p1c_C1.y_test))
print(p1c_C2.model.predict_proba(p1c_C2.X_test)[:5,:])
print(p1c_C2.model.score(p1c_C2.X_test,p1c_C2.y_test))
print(p1c_C3.model.predict_proba(p1c_C3.X_test)[:5,:])
print(p1c_C3.model.score(p1c_C3.X_test,p1c_C3.y_test))

# for C in C_values:
#     pass  #  train and store model in models_p2[C]


# In[84]:


# Side-by-side decision boundary plots
#  create a 1x3 subplot, one per C value
#       each subplot: scatter + decision boundary

print(p1c_C1.true_intercept)
print(p1c_C1.true_coefficients)
x2_vals_C1 = -(p1c_C1.true_intercept[0] + (p1c_C1.true_coefficients[0][0] * np.array(p1c_C1.X[p1c_C1.feature_names[0]]))) / p1c_C1.true_coefficients[0][1]

print(p1c_C2.true_intercept)
print(p1c_C2.true_coefficients)
x2_vals_C2 = -(p1c_C2.true_intercept[0] + (p1c_C2.true_coefficients[0][0] * np.array(p1c_C2.X[p1c_C2.feature_names[0]]))) / p1c_C2.true_coefficients[0][1]

print(p1c_C3.true_intercept)
print(p1c_C3.true_coefficients)
x2_vals_C3 = -(p1c_C3.true_intercept[0] + (p1c_C3.true_coefficients[0][0] * np.array(p1c_C3.X[p1c_C3.feature_names[0]]))) / p1c_C3.true_coefficients[0][1]

# plt.scatter(label0_df_p1c[c_names_df_p1c[0]],label0_df_p1c[c_names_df_p1c[1]],label="Label 0",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label1_df_p1c[c_names_df_p1c[0]],label1_df_p1c[c_names_df_p1c[1]],label="Label 1",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Feature 2 vs. Feature 1")
# plt.grid()
# plt.legend()
# plt.show()

p1c_C1.calculate_label_bmask_clustering(p1c_C1.y)
p1c_C1.plot_clustering(plt,p1c_C1.X,0,1,label_colors=["red","blue","green","orange"])

plt.plot(p1c_C1.X[p1c_C1.feature_names[0]],x2_vals_C1,label="Decision Boundary C = {}".format(C_values[0]),color="black")
plt.plot(p1c_C2.X[p1c_C2.feature_names[0]],x2_vals_C2,label="Decision Boundary C = {}".format(C_values[1]),color="brown")
plt.plot(p1c_C3.X[p1c_C3.feature_names[0]],x2_vals_C3,label="Decision Boundary C = {}".format(C_values[2]),color="orange")
plt.legend()
p1c_C1.plot_labels_show("Feature 1","Feature 2","Feature 2 vs. Feature 1")

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# print(p1c_C1.model_prediction)
p1c_C1.calculate_label_bmask_clustering(p1c_C1.y)
p1c_C1.plot_clustering(ax1,p1c_C1.X,0,1,label_colors=["red","blue"],legend_plt="no")
p1c_C1.calculate_label_bmask_clustering(p1c_C1.model_prediction)
p1c_C1.plot_clustering(ax1,p1c_C1.X_test,0,1,label_colors=["green","orange"],label_labels=["Test Label 1","Test Label 2"],legend_plt="no")
ax1.plot(p1c_C1.X[p1c_C1.feature_names[0]],x2_vals_C1,label="Decision Boundary C = {}".format(C_values[0]),color="black")
ax1.legend()

p1c_C2.calculate_label_bmask_clustering(p1c_C2.y)
p1c_C2.plot_clustering(ax2,p1c_C2.X,0,1,label_colors=["red","blue"],legend_plt="no")
p1c_C2.calculate_label_bmask_clustering(p1c_C2.model_prediction)
p1c_C2.plot_clustering(ax2,p1c_C2.X_test,0,1,label_colors=["green","orange"],label_labels=["Test Label 1","Test Label 2"],legend_plt="no")
ax2.plot(p1c_C2.X[p1c_C2.feature_names[0]],x2_vals_C2,label="Decision Boundary C = {}".format(C_values[1]),color="brown")
ax2.legend()

p1c_C3.calculate_label_bmask_clustering(p1c_C3.y)
p1c_C3.plot_clustering(ax3,p1c_C3.X,0,1,label_colors=["red","blue"],legend_plt="no")
p1c_C3.calculate_label_bmask_clustering(p1c_C3.model_prediction)
p1c_C3.plot_clustering(ax3,p1c_C3.X_test,0,1,label_colors=["green","orange"],label_labels=["Test Label 1","Test Label 2"],legend_plt="no")
ax3.plot(p1c_C3.X[p1c_C3.feature_names[0]],x2_vals_C3,label="Decision Boundary C = {}".format(C_values[2]),color="pink")
ax3.legend()
# p1c_C1.plot_labels_show("Fuel Flow Normalized","Thrust Normalized","Thrust Normalized vs. Fuel Flow Normalized")
p1c_C1.plot_labels_show_sub(fig,"Feature 1","Feature 2","Feature 2 vs. Feature 1")

# fig, (ax1, ax2) = plt.subplots(1, 2)
# p3a.calculate_label_bmask_clustering(p3a.y)
# p3a.plot_clustering(ax1,p3a.X,0,1,label_colors=["red","blue","green"])
# p3a.plot_clusters_same(ax1,0,1)
# p3a.calculate_label_bmask_clustering(p3a.labels)
# p3a.plot_clustering(ax2,p3a.X,0,1,label_colors=["purple","yellow","gray"])
# p3a.plot_clusters_same(ax2,0,1)
# p3a.plot_labels_show_sub(fig,"Fuel Flow Normalized","Thrust Normalized","Thrust Normalized vs. Fuel Flow Normalized")


# In[85]:


# Metrics table
#  for each C value, print accuracy, precision, recall, F1 on the test set
#       format as a table (e.g., a DataFrame)

p1c_C1.key_metrics_classifiers()
p1c_C1.print_key_metrics_classifers()
print(pd.DataFrame(p1c_C1.confusion_matrix,index=['Actual Feature 1','Actual Feature 2'],columns=['Predicted Feature 1','Predicted Feature 2']))
print()

p1c_C2.key_metrics_classifiers()
p1c_C2.print_key_metrics_classifers()
print(pd.DataFrame(p1c_C2.confusion_matrix,index=['Actual Feature 1','Actual Feature 2'],columns=['Predicted Feature 1','Predicted Feature 2']))
print()

p1c_C3.key_metrics_classifiers()
p1c_C3.print_key_metrics_classifers()
print(pd.DataFrame(p1c_C3.confusion_matrix,index=['Actual Feature 1','Actual Feature 2'],columns=['Predicted Feature 1','Predicted Feature 2']))
print()


# **Part 2 Analysis:**
# 
# **Decision boundary vs. C:** 
# 
# As C is changed for the logistic regression, the slope and intercept of the decision boundary changes slightly relative to the orders of magnitude increase in the value of C.
# 
# **Best C:** [Your choice and justification]
# 
# The best C is likely C=0.1. The data is inherently overlapping in many respects and getting a decision boundary that actually correctly predicts every point is impossible without more advanced methods and more information. Therefore, C=0.1 was chosen purely because it guesses Label 2 points ever so slightly better than the other values of C. 
# 
# **Mathematical effect of regularization on the objective:** 
# 
# Decreasing the value of C inversely increases the effect strength of regularization. So it seems like adding more regularization is limiting the effect of overfitting even though its not super obvious in such a simple problem. The default for logistic regression is L2, which is equal to the sum of the squared coefficients and works well with multicollinearity and when all the features are useful. From the results shown I feel like adding the extra penalty term whether its L1 or L2 is a good idea.

# ---
# ## Part 3 – $k$-Means Clustering

# ### 3a — Well-Behaved Clustering

# In[86]:


# Load operating mode data
#  load data/p2a_operating_modes.csv
file_str_p3a = "p2a_operating_modes.csv"
df_p3a = pd.read_csv("{}\\{}".format(directory_str,file_str_p3a))
print(df_p3a.head(10))

df_p3a_summary_stats = df_p3a.describe()
c_names_df_p3a = df_p3a.columns
numr_df_p3a = df_p3a[c_names_df_p3a[0]].value_counts()
print(df_p3a_summary_stats)

# Select only the two feature columns for clustering
# X_p3a = None
X = df_p3a[c_names_df_p3a[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p3a[c_names_df_p3a[-1]]  # label column
p3a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p3a.custom_X(X)

# Apply k-means with k=3
# #  KMeans(n_clusters=3, random_state=42)
# km_p3a = None
p3a.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":3})


# In[87]:


# Scatter plot: discovered clusters + centroids
#  color points by km_p3a.labels_, mark centroids with a different marker

# print(df_p3a[c_names_df_p3a[2]])
# label0_df_p3a = df_p3a[df_p3a[c_names_df_p3a[2]] == 0].copy()
# label1_df_p3a = df_p3a[df_p3a[c_names_df_p3a[2]] == 1].copy()
# label2_df_p3a = df_p3a[df_p3a[c_names_df_p3a[2]] == 2].copy()
# plt.scatter(label0_df_p3a[c_names_df_p3a[0]],label0_df_p3a[c_names_df_p3a[1]],label="Label 1",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label1_df_p3a[c_names_df_p3a[0]],label1_df_p3a[c_names_df_p3a[1]],label="Label 2",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label2_df_p3a[c_names_df_p3a[0]],label2_df_p3a[c_names_df_p3a[1]],label="Label 3",color="green") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(p3a.cluster_centers_unscaled[0][0],p3a.cluster_centers_unscaled[0][1],s=100,marker="*",label="Cluster 1",color="brown",edgecolors="black")
# plt.scatter(p3a.cluster_centers_unscaled[1][0],p3a.cluster_centers_unscaled[1][1],s=100,marker="*",label="Cluster 2",color="pink",edgecolors="black")
# plt.scatter(p3a.cluster_centers_unscaled[2][0],p3a.cluster_centers_unscaled[2][1],s=100,marker="*",label="Cluster 3",color="orange",edgecolors="black")
# plt.xlabel("Fuel Flow Normalized")
# plt.ylabel("Thrust Normalized")
# plt.title("Thrust Normalized vs. Fuel Flow Normalized")
# plt.grid()
# plt.legend()
# plt.show()

p3a.calculate_label_bmask_clustering(p3a.y)
p3a.plot_clustering(plt,p3a.X,0,1,label_colors=["red","blue","green","orange"])
p3a.plot_clusters_same(plt,0,1)
p3a.plot_labels_show("Fuel Flow Normalized","Thrust Normalized","Thrust Normalized vs. Fuel Flow Normalized")


# In[88]:


# Silhouette score
#  compute and print silhouette_score(X_p3a, km_p3a.labels_)

p3a.key_metrics_clustering()
p3a.print_key_metrics_clustering()


# In[89]:


# Compare with true labels
#  side-by-side scatter: left = k-means labels, right = true_mode column

# p3a_results = pd.merge(pd.DataFrame(p3a.model_prediction,columns=p3a.)
# df_p3a_test_values = df_p3a.iloc[p3a.test_indices].copy()
# p3a_result_xs = p3a.data.iloc[p3a.test_indices][c_names_df_p3a[:-1]]
# p3a_result_ys = p3a.data.iloc[p3a.test_indices][c_names_df_p3a[-1]]
# print(df_test_p3a[c_names_df_p3a[0]])
# print([p3a.y_test == 0])
# print(type(p3a.X_train))
# print(type(p3a.y_test))
# print(type(np.array(p3a.model_prediction == 0)))
# print(p3a.model_prediction)
# print(np.array(p3a.model_prediction == 0))
# print(p3a.test_indices)
# print(np.array(p3a.test_indices)[np.array(p3a.model_prediction == 0)])
# print(p3a.labels[p3a.data[c_names_df_p3a[-1]] == 0].copy() )
# print(p3a.labels[c_names_df_p3a[-1]])
# print([p3a.labels[c_names_df_p3a[-1]] == 0])
# df_test_p3a = df_p3a.iloc[p3a.test_indices].copy()
# # print(df_p3a[c_names_df_p3a[-1]], [df_p3a.iloc[p3a.test_indices][c_names_df_p3a[-1]] == 0])



# label0_df_p3a_test_x = df_p3a[c_names_df_p3a[0]][p3a.labels == 0] # df_test_p3a[c_names_df_p3a[0]][np.array(p3a.test_indices)[np.array(p3a.model_prediction == 0)]] # df_test_p3a[c_names_df_p3a[0]][p3a.y_test == 0]
# label1_df_p3a_test_x = df_p3a[c_names_df_p3a[0]][p3a.labels == 1] # df_test_p3a[c_names_df_p3a[0]][np.array(p3a.test_indices)[np.array(p3a.model_prediction == 1)]] # df_test_p3a[c_names_df_p3a[0]][p3a.y_test == 1]
# label2_df_p3a_test_x = df_p3a[c_names_df_p3a[0]][p3a.labels == 2] # df_test_p3a[c_names_df_p3a[0]][np.array(p3a.test_indices)[np.array(p3a.model_prediction == 2)]] # df_test_p3a[c_names_df_p3a[0]][p3a.y_test == 2]
# label0_df_p3a_test_y = df_p3a[c_names_df_p3a[1]][p3a.labels == 0] # df_test_p3a[c_names_df_p3a[1]][np.array(p3a.test_indices)[np.array(p3a.model_prediction == 0)]] # df_test_p3a[c_names_df_p3a[1]][p3a.model_prediction == 0]
# label1_df_p3a_test_y = df_p3a[c_names_df_p3a[1]][p3a.labels == 1] # df_test_p3a[c_names_df_p3a[1]][np.array(p3a.test_indices)[np.array(p3a.model_prediction == 1)]] # df_test_p3a[c_names_df_p3a[1]][p3a.model_prediction == 1]
# label2_df_p3a_test_y = df_p3a[c_names_df_p3a[1]][p3a.labels == 2] # df_test_p3a[c_names_df_p3a[1]][np.array(p3a.test_indices)[np.array(p3a.model_prediction == 2)]] # df_test_p3a[c_names_df_p3a[1]][p3a.model_prediction == 2]

# # print(label0_df_p3a_test_x)
# # print(label0_df_p3a_test_y)
# fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1.scatter(label0_df_p3a[c_names_df_p3a[0]],label0_df_p3a[c_names_df_p3a[1]],label="Original Label 1",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# ax1.scatter(label1_df_p3a[c_names_df_p3a[0]],label1_df_p3a[c_names_df_p3a[1]],label="Original Label 2",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# ax1.scatter(label2_df_p3a[c_names_df_p3a[0]],label2_df_p3a[c_names_df_p3a[1]],label="Original Label 3",color="green") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# ax1.scatter(p3a.cluster_centers_unscaled[0][0],p3a.cluster_centers_unscaled[0][1],s=100,marker="*",label="Cluster 1",color="brown",edgecolors="black")
# ax1.scatter(p3a.cluster_centers_unscaled[1][0],p3a.cluster_centers_unscaled[1][1],s=100,marker="*",label="Cluster 2",color="pink",edgecolors="black")
# ax1.scatter(p3a.cluster_centers_unscaled[2][0],p3a.cluster_centers_unscaled[2][1],s=100,marker="*",label="Cluster 3",color="orange",edgecolors="black")
# # ax1.legend()

# ax2.scatter(label0_df_p3a_test_x,label0_df_p3a_test_y,label="Predicted Label 1",color="purple") # color=(np.random.random(), np.random.random(), np.random.random()))
# ax2.scatter(label1_df_p3a_test_x,label1_df_p3a_test_y,label="Predicted Label 2",color="yellow") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# ax2.scatter(label2_df_p3a_test_x,label2_df_p3a_test_y,label="Predicted Label 3",color="gray") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# ax2.scatter(p3a.cluster_centers_unscaled[0][0],p3a.cluster_centers_unscaled[0][1],s=100,marker="*",label="Cluster 1",color="brown",edgecolors="black")
# ax2.scatter(p3a.cluster_centers_unscaled[1][0],p3a.cluster_centers_unscaled[1][1],s=100,marker="*",label="Cluster 2",color="pink",edgecolors="black")
# ax2.scatter(p3a.cluster_centers_unscaled[2][0],p3a.cluster_centers_unscaled[2][1],s=100,marker="*",label="Cluster 3",color="orange",edgecolors="black")
# # ax2.legend()
# fig.supxlabel("Fuel Flow Normalized")
# fig.supylabel("Thrust Normalized")
# fig.suptitle("Thrust Normalized vs. Fuel Flow Normalized")
# plt.show()

# plt.scatter(label0_df_p3a[c_names_df_p3a[0]],label0_df_p3a[c_names_df_p3a[1]],label="Original Label 1",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label1_df_p3a[c_names_df_p3a[0]],label1_df_p3a[c_names_df_p3a[1]],label="Original Label 2",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label2_df_p3a[c_names_df_p3a[0]],label2_df_p3a[c_names_df_p3a[1]],label="Original Label 3",color="green") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# plt.scatter(label0_df_p3a_test_x,label0_df_p3a_test_y,label="Predicted Label 1",color="purple") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label1_df_p3a_test_x,label1_df_p3a_test_y,label="Predicted Label 2",color="yellow") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label2_df_p3a_test_x,label2_df_p3a_test_y,label="Predicted Label 3",color="gray") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# plt.scatter(p3a.cluster_centers_unscaled[0][0],p3a.cluster_centers_unscaled[0][1],s=100,marker="*",label="Cluster 1",color="brown",edgecolors="black")
# plt.scatter(p3a.cluster_centers_unscaled[1][0],p3a.cluster_centers_unscaled[1][1],s=100,marker="*",label="Cluster 2",color="pink",edgecolors="black")
# plt.scatter(p3a.cluster_centers_unscaled[2][0],p3a.cluster_centers_unscaled[2][1],s=100,marker="*",label="Cluster 3",color="orange",edgecolors="black")
# plt.xlabel("Fuel Flow Normalized")
# plt.ylabel("Thrust Normalized")
# plt.title("Thrust Normalized vs. Fuel Flow Normalized")
# plt.grid()
# plt.legend()
# plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2)
p3a.calculate_label_bmask_clustering(p3a.y)
p3a.plot_clustering(ax1,p3a.X,0,1,label_colors=["red","blue","green"])
p3a.plot_clusters_same(ax1,0,1)
p3a.calculate_label_bmask_clustering(p3a.labels)
p3a.plot_clustering(ax2,p3a.X,0,1,label_colors=["purple","yellow","gray"])
p3a.plot_clusters_same(ax2,0,1)
p3a.plot_labels_show_sub(fig,"Fuel Flow Normalized","Thrust Normalized","Thrust Normalized vs. Fuel Flow Normalized")

p3a.calculate_label_bmask_clustering(p3a.y)
p3a.plot_clustering(plt,p3a.X,0,1,label_colors=["red","blue","green"])
p3a.plot_clusters_same(plt,0,1)
p3a.calculate_label_bmask_clustering(p3a.labels)
p3a.plot_clustering(plt,p3a.X,0,1,label_colors=["purple","yellow","gray"])
p3a.plot_clusters_same(plt,0,1)
p3a.plot_labels_show("Fuel Flow Normalized","Thrust Normalized","Thrust Normalized vs. Fuel Flow Normalized")


# **3a Analysis:**
# 
# [Your answer here: Under what conditions does k-means work well? Do those conditions hold here?]
# 
# K-means clustering work very well when searching for classification for future labels. In the case of this problem, the labels are already provided. Additionally, the data is very clearly clustered in areas very far away from each other, which makes it an ideal case for k-means clustering as no data will be mistaken by the model for another label.

# ### 3b — Choosing $k$

# In[90]:


# Load choose-k dataset
#  load data/p2b_choose_k.csv
file_str_p3b = "p2b_choose_k.csv"
df_p3b = pd.read_csv("{}\\{}".format(directory_str,file_str_p3b))
print(df_p3b.head(10))

df_p3b_summary_stats = df_p3b.describe()
c_names_df_p3b = df_p3b.columns
numr_df_p3b = df_p3b[c_names_df_p3b[0]].value_counts()
print(df_p3b_summary_stats)

X = df_p3b[c_names_df_p3b[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p3b[c_names_df_p3b[-1]]  # label column

# Run k-means for k = 1..10, record WCSS
k_range = range(1, 11)
wcss = []
sil_scores = []

for k in k_range:
    p3b = ModelML("K-Clustering",X,random_state = 42) # "LogisticRegression", # y
    p3b.custom_X(X)
    p3b.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":k})
    print(k)
    # print(np.shape(p3b.X_test))
    # print(np.shape(p3b.model_prediction))
    # print(p3b.model_prediction)
    # print(p3b.y_test)
    # print(p3b.y_train)
    # print(p3b.labels)
    if k != 1:
        p3b.key_metrics_clustering()
        sil_scores.append(p3b.silhouette_score)
    else:
        sil_scores.append(0)
    wcss.append(p3b.WCSS)

    # pass  #  fit KMeans, append inertia_ to wcss


# In[91]:


# Elbow plot
#  plot WCSS vs k, mark the elbow point

plt.plot(k_range,wcss,color="red")
plt.scatter(k_range[3],wcss[3],label="Elbow Point",color="blue")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.title("WCSS vs. k")
plt.grid()
plt.legend()
plt.show()


# In[92]:


# Silhouette scores for k = 2..10


# for k in range(2, 11):
#     pass  #  fit KMeans, compute silhouette_score, append

#  plot silhouette score vs k, mark maximum

plt.plot(k_range,sil_scores,color="red")
plt.scatter(k_range[3],sil_scores[3],label="Elbow Point",color="blue")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. k")
plt.grid()
plt.legend()
plt.show()


# In[93]:


# Final clustering with chosen k
best_k = 4  #  set your chosen k
#  fit and plot the final clustering

p3b = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p3b.custom_X(X)
p3b.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":best_k})

# print(df_p3b[c_names_df_p3b[2]])
# label0_df_p3b = df_p3b[df_p3b[c_names_df_p3b[2]] == 0].copy()
# label1_df_p3b = df_p3b[df_p3b[c_names_df_p3b[2]] == 1].copy()
# label2_df_p3b = df_p3b[df_p3b[c_names_df_p3b[2]] == 2].copy()
# label3_df_p3b = df_p3b[df_p3b[c_names_df_p3b[2]] == 3].copy()

# plt.scatter(label0_df_p3b[c_names_df_p3b[0]],label0_df_p3b[c_names_df_p3b[1]],label="Label 1",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label1_df_p3b[c_names_df_p3b[0]],label1_df_p3b[c_names_df_p3b[1]],label="Label 2",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label2_df_p3b[c_names_df_p3b[0]],label2_df_p3b[c_names_df_p3b[1]],label="Label 3",color="green") # ,color=(np.random.random(), np.random.random(), np.random.random()))
# plt.scatter(label3_df_p3b[c_names_df_p3b[0]],label3_df_p3b[c_names_df_p3b[1]],label="Label 4",color="orange") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# # plt.scatter(p3a.cluster_centers[0][0],p3a.cluster_centers[0][1],s=100,marker="*",label="Cluster 1",color="brown",edgecolors="black")
# # plt.scatter(p3a.cluster_centers[1][0],p3a.cluster_centers[1][1],s=100,marker="*",label="Cluster 2",color="pink",edgecolors="black")
# # plt.scatter(p3a.cluster_centers[2][0],p3a.cluster_centers[2][1],s=100,marker="*",label="Cluster 3",color="orange",edgecolors="black")
# plt.scatter(p3b.cluster_centers_unscaled[0][0],p3b.cluster_centers_unscaled[0][1],s=200,marker="*",color="black",edgecolors="white")
# plt.scatter(p3b.cluster_centers_unscaled[1][0],p3b.cluster_centers_unscaled[1][1],s=200,marker="*",color="black",edgecolors="white")
# plt.scatter(p3b.cluster_centers_unscaled[2][0],p3b.cluster_centers_unscaled[2][1],s=200,marker="*",color="black",edgecolors="white")
# plt.scatter(p3b.cluster_centers_unscaled[3][0],p3b.cluster_centers_unscaled[3][1],s=200,marker="*",color="black",edgecolors="white")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("x2 vs. x1")
# plt.grid()
# plt.legend()
# plt.show()

# print(type(p3b.cluster_centers_unscaled))

p3b.calculate_label_bmask_clustering(p3b.y)
p3b.plot_clustering(plt,p3b.X,0,1,label_colors=["red","blue","green","orange"])
p3b.plot_clusters_same(plt,0,1)
p3b.plot_labels_show("x1","x2","x2 vs. x1")


# **3b Analysis:**
# 
# **Chosen k:** [state your k]
# 
# The chosen k is 4. 
# 
# **Justification:** [Your answer here: do both criteria agree? Which do you trust more, and why?]
# 
# This is because the graph of Within-Cluster Sum of Squares (WCSS) vs. k plot clearly shows a massive drop in WCSS as k=4. This indicates that at 4 points the sum of squares between the centroid of each cluster and nearby points drops significantly. This is called the Elbow Point and indicates an optimal set of clusters. This is supported by the Silhouette Score at k=4, which reaches a peak. A Silhouette Score of greater than 0.7, as is shown for k=4 is considered a strong structure. The Silhouette Score is measures cluster proximity using the mean distance of same cluster points and the mean distance to other clusters. I trust the Silhouette Score more because there is very clearly a peak at k=4. The WCSS could arguably be misinterpreted to be equal to 5 or 6 depending on user interpretation as past the Elbow Point the WCSS does slightly continue to decrease.

# ### 3c — The Importance of Feature Scaling

# In[94]:


# Load scaling dataset
#  load data/p2c_scaling.csv
file_str_p3c = "p2c_scaling.csv"
df_p3c = pd.read_csv("{}\\{}".format(directory_str,file_str_p3c))

df_p3c_summary_stats = df_p3c.describe()
c_names_df_p3c = df_p3c.columns
numr_df_p3c = df_p3c[c_names_df_p3c[0]].value_counts()
print(df_p3c_summary_stats)


# In[95]:


# K-means WITHOUT standardization
#  KMeans(n_clusters=2, random_state=42) on raw X_p3c
X = df_p3c[c_names_df_p3c[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p3c[c_names_df_p3c[-1]]  # label column

p3c_raw = ModelML("K-Clustering",X,y,random_state = 42)
p3c_raw.custom_X(X)
p3c_raw.full_model_process_clustering(scale = "no",kwargs_model={"max_iter":1000,"n_clusters":2})

# K-means WITH standardization
#  StandardScaler -> transform -> KMeans(n_clusters=2, random_state=42)
p3c_std = ModelML("K-Clustering",X,y,random_state = 42)
p3c_std.custom_X(X)
p3c_std.full_model_process_clustering(scale = "yes",kwargs_model={"max_iter":1000,"n_clusters":2})


# In[96]:


# Side-by-side scatter plots: raw vs scaled clustering
#  1x2 subplot; left = raw, right = scaled; both in original feature space

# label0_df_p3c_raw = df_p3c[df_p3c[c_names_df_p3c[2]] == 0].copy()
# label1_df_p3c_raw = df_p3c[df_p3c[c_names_df_p3c[2]] == 1].copy()
# label0_df_p3c_std = p3c_std.X_main[df_p3c[c_names_df_p3c[2]] == 0].copy()
# label1_df_p3c_std = p3c_std.X_main[df_p3c[c_names_df_p3c[2]] == 1].copy()
# # print(label0_df_p3c_std)
# print(type(label0_df_p3c_std))
# print(p3c_std.cluster_centers)

# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False)

# ax1.scatter(label0_df_p3c_raw[c_names_df_p3c[0]],label0_df_p3c_raw[c_names_df_p3c[1]],label="Unscaled Label 1",color="red") # color=(np.random.random(), np.random.random(), np.random.random()))
# ax1.scatter(label1_df_p3c_raw[c_names_df_p3c[0]],label1_df_p3c_raw[c_names_df_p3c[1]],label="Unscaled Label 2",color="blue") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# ax1.scatter(p3c_raw.cluster_centers[0][0],p3c_raw.cluster_centers[0][1],s=200,marker="*",label="Cluster 1",color="black",edgecolors="white")
# ax1.scatter(p3c_raw.cluster_centers[1][0],p3c_raw.cluster_centers[1][1],s=200,marker="*",label="Cluster 2",color="black",edgecolors="white")
# # ax1.legend()

# ax2.scatter(label0_df_p3c_std[:,0],label0_df_p3c_std[:,1],label="Scaled Label 1",color="purple") # color=(np.random.random(), np.random.random(), np.random.random()))
# ax2.scatter(label1_df_p3c_std[:,0],label1_df_p3c_std[:,1],label="Scaled Label 2",color="yellow") # ,color=(np.random.random(), np.random.random(), np.random.random()))

# ax2.scatter(p3c_std.cluster_centers[0][0],p3c_std.cluster_centers[0][1],s=200,marker="*",label="Cluster 1",color="black",edgecolors="white")
# ax2.scatter(p3c_std.cluster_centers[1][0],p3c_std.cluster_centers[1][1],s=200,marker="*",label="Cluster 2",color="black",edgecolors="white")

# fig.supxlabel("Temperature (C)")
# fig.supylabel("Oil Pressure (Psi)")
# fig.suptitle("Oil Pressure vs. Temperature")

fig, (ax1, ax2) = plt.subplots(1, 2)
p3c_raw.calculate_label_bmask_clustering(p3c_raw.y)
p3c_raw.plot_clustering(ax1,p3c_raw.X,0,1,label_colors=["red","blue","green"])
p3c_raw.plot_clusters_same(ax1,0,1)
p3c_std.calculate_label_bmask_clustering(p3c_std.labels)
p3c_std.plot_clustering(ax2,p3c_std.X,0,1,label_colors=["purple","yellow","gray"])
p3c_std.plot_clusters_same(ax2,0,1)
p3c_std.plot_labels_show_sub(fig,"Temperature (C)","Oil Pressure (Psi)","Oil Pressure (Psi) vs. Temperature (C)")


# In[97]:


# Silhouette scores and ARI for both runs
# true_labels = df_p3c['true_state'].values
p3c_raw.key_metrics_clustering()
p3c_raw.print_key_metrics_clustering()
p3c_std.key_metrics_clustering()
p3c_std.print_key_metrics_clustering()


#  compute silhouette_score and adjusted_rand_score for km_raw
#  compute silhouette_score and adjusted_rand_score for km_scaled
#  print as a comparison table


# **3c Analysis:**
# 
# [Your answer here: Why is k-means sensitive to feature scale? Why does Euclidean distance create this sensitivity? Should you always standardize?]
# 
# K-means clusters according to Euclidean Distance, which is the sum of the squares of the change in variables square rooted. This naturally, due to the nature of squaring terms, leads to larger feature differences dominating over smaller feature differences. This means in most scenarios you should standardize according to z-score like StandardScaler does. You should always standardize as the additional computational overhead may not be worth it or you could potentially take advantage of the dominance of larger terms in the normal basis to shape the model in the way you want. Although, generally its almost always better to standardize as neither of these issues or use cases are ever that important.

# ---
# # Part 2: Failure Modes and Limits 
# > 

# ---
# ## Part 4 – When Logistic Regression Fails

# ### 4a — Non-Linear Decision Boundary

# In[98]:


# Load concentric-ring dataset
#  load data/p3a_nonlinear.csv
file_str_p4a = "p3a_nonlinear.csv"
df_p4a = pd.read_csv("{}\\{}".format(directory_str,file_str_p4a))
print(df_p4a.head(10))

df_p4a_summary_stats = df_p4a.describe()
c_names_df_p4a = df_p4a.columns
numr_df_p4a = df_p4a[c_names_df_p4a[0]].value_counts()
print(df_p4a_summary_stats)

X = df_p4a[c_names_df_p4a[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p4a[c_names_df_p4a[-1]]  # label column

p4a = ModelML("LogisticRegression",X,y,random_state = 42) # "LogisticRegression",
p4a.custom_X(X)
p4a.full_model_process(kwargs_model={"max_iter":1000})

# Scatter plot colored by label
#  scatter plot confirming non-linear separability
p4a.calculate_label_bmask_clustering(p4a.y)
p4a.plot_clustering(plt,p4a.X,0,1,label_colors=["red","blue","green","orange"])
p4a.plot_labels_show("x1","x2","x2 vs. x1")
# p4a.plot_clustering(plt,p4a.features,0,1)


# In[99]:


# Standard logistic regression (no feature engineering)
#  80/20 split, train LogisticRegression(max_iter=1000)
# clf_p4a_linear = None

#  report test accuracy
p4a.key_metrics_classifiers()
p4a.print_key_metrics_classifers
#  plot linear decision boundary (show it fails to separate the rings)

p4a.plot_clustering(plt,p4a.X,0,1,label_colors=["red","blue","green","orange"])
p4a.decision_boundary_values(0,1)
p4a.plot_labels_show("x1","x2","x2 vs. x1")


# In[100]:


# Feature engineering: add x1^2 and x2^2
#  create X_engineered with columns [x1, x2, x1^2, x2^2]
# X_engineered = None

X_mod = X.copy()
# print(np.square(X[c_names_df_p4a[0]]).shape[0])
X_mod["x1^2"] = np.square(X[c_names_df_p4a[0]])
X_mod["x2^2"] = np.square(X[c_names_df_p4a[1]])
# print(X_mod)
p4a = ModelML("LogisticRegression",X_mod,y,random_state = 42) # "LogisticRegression",
p4a.custom_X(X_mod)
# print(p4a.features)
p4a.full_model_process(kwargs_model={"max_iter":1000})

#  re-split with the same random_state=42
#  train LogisticRegression(max_iter=1000) on engineered features
clf_p4a_eng = None

#  report improved test accuracy
#  report test accuracy
p4a.key_metrics_classifiers()
p4a.print_key_metrics_classifers
#  plot linear decision boundary (show it fails to separate the rings)

p4a.calculate_label_bmask_clustering(p4a.y)
p4a.plot_clustering(plt,p4a.X,0,1,label_colors=["red","blue","green","orange"])
p4a.decision_boundary_values(0,1)
p4a.plot_decision_boundary()
plt.legend()
p4a.plot_labels_show("x1","x2","x2 vs. x1")


# In[101]:


# Plot decision boundary in original x1/x2 space
#  meshgrid over x1, x2 -> add squared features -> predict -> contourf
#       overlay original data points

p4a.calculate_label_bmask_clustering(p4a.y)
p4a.grid_decision_boundary_values(0,1)
p4a.plot_decision_boundary_contour()
p4a.plot_clustering(plt,p4a.X,0,1,label_colors=["red","blue","green","orange"])
# plt.xlim(-2,2)
# plt.ylim(0,2)
plt.legend()
p4a.plot_labels_show("x1","x2","x2 vs. x1")


# **4a Analysis:**
# 
# **Why adding x1² and x2² works:** 
# 
# The equation for the radius of a circle is a function of x^2 and y^2, so it makes sense that this would create a circular boundary. 
# 
# **Can you always find such transformations?** 
# 
# No, not always. There are many more functions and relationships between variables that are possible. In many cases, this will necessitate numerical root finding methods for more complex equations.
# 
# **What to use when the boundary structure is unknown:** 
# 
# Without knowing the boundary structure other more advanced classifier algorithms would have to be used or an unsupervised algorithm like K-means that doesn’t require knowledge of specific functions.

# ### 4b — Gradient Descent from Scratch

# In[102]:


# Use the p1a dataset; standardize features and add bias column
#  load p1a_sensor_binary.csv (or reuse df_p1)
#  StandardScaler on features
#  add bias column of ones -> X_bias
file_str_p4b = "p1a_sensor_binary.csv"
df_p4b = pd.read_csv("{}\\{}".format(directory_str,file_str_p4a))
print(df_p4b.head(4))

df_p4b_summary_stats = df_p4b.describe()
c_names_df_p4b = df_p4b.columns
numr_df_p4b = df_p4b[c_names_df_p4a[0]].value_counts()
print(df_p4b_summary_stats)

X = df_p4b[c_names_df_p4b[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p4b[c_names_df_p4b[-1]]  # label column

# X.insert(0,"Bias",np.ones(len(numr_df_p4b)))
# print(X)
p4b_bias = ModelML("LogisticRegression",X,y,random_state = 42) # "LogisticRegression",
p4b_bias.custom_X(X)
p4b_bias.tt_split()
p4b_bias.sd_scale()
# print(len(p4b.X_train[p4b.feature_names[0]]))
p4b_bias.set_main_model_Xy()
p4b_bias.X_train_main.insert(0,"Bias",np.ones(len(p4b_bias.X_train_main[p4b_bias.feature_names[0]])))
p4b_bias.X_test_main.insert(0,"Bias",np.ones(len(p4b_bias.X_test_main[p4b_bias.feature_names[0]])))
p4b_bias.create_model(kwargs_model={"max_iter":1000,"fit_intercept":False})
p4b_bias.predict_model()
p4b_bias.coefficients_model(biased="yes")
p4b_bias.cnvrt_to_real_coefficient_model()

print(p4b_bias.X_train.head(4))
print(p4b_bias.X_train_main.head(4))
print(p4b_bias.intercept)
print(p4b_bias.coefficients)
print(p4b_bias.true_intercept)
print(p4b_bias.true_coefficients)

p4b = ModelML("LogisticRegression",X,y,random_state = 42) # "LogisticRegression",
p4b.custom_X(X)
p4b.full_model_process()

# Scatter plot colored by label
#  scatter plot confirming non-linear separability

p4b.calculate_label_bmask_clustering(p4b.y)
p4b.plot_clustering(plt,p4b.X,0,1,label_colors=["red","blue"],s=150,label_labels=["Default Label 1","Default Label 2"])
p4b_bias.calculate_label_bmask_clustering(p4b_bias.y)
p4b_bias.plot_clustering(plt,p4b_bias.X,0,1,label_colors=["green","orange"],s=100,label_labels=["Biased Label 1","Biased Label 2"])
p4b_bias.plot_labels_show("x1","x2","x2 vs. x1")


# In[103]:


# ── Implement the three functions below ────────────────────────────────────

# def sigmoid(z):
#     """Element-wise sigmoid function."""
#     #  implement
#     pass


# def cross_entropy(X, y, beta):
#     """
#     Binary cross-entropy loss.
#     J(beta) = -1/n * sum[ y*log(p) + (1-y)*log(1-p) ]
#     """
#     #  implement
#     pass


# def gradient(X, y, beta):
#     """
#     Gradient of cross-entropy w.r.t. beta.
#     grad = 1/n * X^T (p - y)
#     """
#     #  implement
#     pass


# def gradient_descent(X, y, lr=0.1, n_iters=1000):
#     """
#     Batch gradient descent for logistic regression.
#     Returns: (beta, cost_history)
#     """
#     beta = np.zeros(X.shape[1])
#     cost_history = []
#     #  implement loop
#     return beta, cost_history


# In[104]:


# Run gradient descent with lr=0.1, n_iters=1000
#  call gradient_descent, store beta and cost_history
iterations = 1000

p4b_grad_GD = ModelML("LogisticRegression",X,y,random_state = 42)
p4b_grad_GD.custom_X(X)
p4b_grad_GD.gradient_descent_log_reg(model="GD",learning_rate=0.1,iterations=iterations)

p4b_grad_SGD = ModelML("LogisticRegression",X,y,random_state = 42)
p4b_grad_SGD.custom_X(X)
p4b_grad_SGD.gradient_descent_log_reg(model="SGD",learning_rate=0.1,iterations=iterations)


# In[105]:


# Plot cost vs iteration
#  plot cost_history; verify convergence
iter_vals = np.linspace(1,iterations,iterations)
plt.plot(iter_vals,p4b_grad_GD.GD_cost,label="Gradient Descent",color="red")
plt.plot(iter_vals,p4b_grad_SGD.GD_cost,label="Stochastic Gradient Descent",color="blue")
plt.legend()
p4b_bias.plot_labels_show("Iterations","Cost J(beta)","Cost vs. Iterations")

plt.plot(np.log(iter_vals),np.log(p4b_grad_GD.GD_cost),label="Gradient Descent",color="red")
plt.plot(np.log(iter_vals),np.log(p4b_grad_SGD.GD_cost),label="Stochastic Gradient Descent",color="blue")
plt.legend()
p4b_bias.plot_labels_show("log(Iterations)","log(Cost J(beta))","Cost vs. Iterations")


# In[106]:


# Compare with sklearn on the same (standardized + bias) data
#  train LogisticRegression(C=1e9, max_iter=1000) to approximate unregularized
#  compare fraction of predictions that agree between your GD and sklearn
p4b_unreg = ModelML("LogisticRegression",X,y,random_state = 42) # "LogisticRegression",
p4b_unreg.custom_X(X)
p4b_unreg.full_model_process(kwargs_model={"C":10**9,"max_iter":1000})

print("Normal Logistic Regression:\nIntercepts = {}\nCoefficients = {}".format(p4b_unreg.intercept,p4b_unreg.coefficients))
print("GD Logistic Regression:\nIntercepts = {}\nCoefficients = {}".format(p4b_grad_GD.intercept,p4b_grad_GD.coefficients))
print("SGD Logistic Regression:\nIntercepts = {}\nCoefficients = {}".format(p4b_grad_SGD.intercept,p4b_grad_SGD.coefficients))
print("The models seem to agree very well, however, the minor differences between\nthe two may come from the lack of an L1 modifier in the GD functions or\njust a lack of iterations.")


# In[107]:


# Effect of learning rate
#  run gradient_descent with lr=1.0 and lr=10.0
#  plot cost histories for all three learning rates on one figure
p4b_grad_GD2 = ModelML("LogisticRegression",X,y,random_state = 42)
p4b_grad_GD2.custom_X(X)
p4b_grad_GD2.gradient_descent_log_reg(model="GD",learning_rate=1,iterations=iterations)

p4b_grad_GD3 = ModelML("LogisticRegression",X,y,random_state = 42)
p4b_grad_GD3.custom_X(X)
p4b_grad_GD3.gradient_descent_log_reg(model="GD",learning_rate=10,iterations=iterations)

p4b_grad_SGD2 = ModelML("LogisticRegression",X,y,random_state = 42)
p4b_grad_SGD2.custom_X(X)
p4b_grad_SGD2.gradient_descent_log_reg(model="SGD",learning_rate=1,iterations=iterations)


# In[108]:


plt.plot(iter_vals,p4b_grad_GD.GD_cost,label="Gradient Descent lr=0.1",color="red")
plt.plot(iter_vals,p4b_grad_GD2.GD_cost,label="Gradient Descent lr=1",color="green")
plt.plot(iter_vals,p4b_grad_GD3.GD_cost,label="Gradient Descent lr=10",color="orange")
plt.plot(iter_vals,p4b_grad_SGD.GD_cost,label="Stochastic Gradient Descent lr=0.1",color="blue")
plt.plot(iter_vals,p4b_grad_SGD2.GD_cost,label="Stochastic Gradient Descent lr=1",color="pink")
plt.legend()
p4b_bias.plot_labels_show("Iterations","Cost J(beta)","Cost vs. Iterations")

plt.plot(np.log10(iter_vals),np.log10(p4b_grad_GD.GD_cost),label="Gradient Descent lr=0.1",color="red")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_GD2.GD_cost),label="Gradient Descent lr=1",color="green")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_GD3.GD_cost),label="Gradient Descent lr=10",color="orange")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_SGD.GD_cost),label="Stochastic Gradient Descent lr=0.1",color="blue")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_SGD2.GD_cost),label="Stochastic Gradient Descent lr=1",color="pink")
plt.legend()
p4b_bias.plot_labels_show("log10(Iterations)","log10(Cost J(beta))","Cost vs. Iterations")

plt.plot(iter_vals,p4b_grad_GD.GD_cost,label="Gradient Descent lr=0.1",color="red")
plt.plot(iter_vals,p4b_grad_GD2.GD_cost,label="Gradient Descent lr=1",color="green")
plt.plot(iter_vals,p4b_grad_SGD.GD_cost,label="Stochastic Gradient Descent lr=0.1",color="blue")
plt.plot(iter_vals,p4b_grad_SGD2.GD_cost,label="Stochastic Gradient Descent lr=1",color="pink")
plt.legend()
p4b_bias.plot_labels_show("Iterations","Cost J(beta)","Cost vs. Iterations")

plt.plot(np.log10(iter_vals),np.log10(p4b_grad_GD.GD_cost),label="Gradient Descent lr=0.1",color="red")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_GD2.GD_cost),label="Gradient Descent lr=1",color="green")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_SGD.GD_cost),label="Stochastic Gradient Descent lr=0.1",color="blue")
plt.plot(np.log10(iter_vals),np.log10(p4b_grad_SGD2.GD_cost),label="Stochastic Gradient Descent lr=1",color="pink")
plt.legend()
p4b_bias.plot_labels_show("log10(Iterations)","log10(Cost J(beta))","Cost vs. Iterations")


# **4b Analysis:**
# 
# **Effect of large learning rate:** 
# 
# Using a large learning rate can make the gradient descent extremely unstable, to the point of it not only not converging correctly but also producing completely unusable results. Seemingly, for this specific scenario, increasing the learning rate to 1 didn’t yield any unfortunate results.
# 
# **Why does cost decrease monotonically (or not)?:** 
# 
# The cost decreases monotonically simply because of the use of a logarithm in the cost. This logarithm occurs in the first place because the algorithm is trying to maximize beta, due to the form of the sigmoid function, which in turn minimizes the logarithm in the cost equation. Because of this minimization effect, the gradient descent decreases monotonically unless the learning rate is too high.
# 
# **Batch GD vs SGD — when to prefer SGD:** 
# 
# SGD is typically better with larger datasets due to lower computational costs and memory costs if implemented correctly. However, it’s also useful because it can escape a local minima due to the change in order of the points processed by the algorithm. Batch GD is not as capable of doing this as the order of points in the algorithm is fixed and it uses the entire dataset to updated the regression coefficients. For this specific scenario, using SGD over Batch GD didn’t yield any unique results for a learning rate=0.1, but a learning rate=1 began to show a noisy effect in the convergence of the cost function which is characteristic of SGD. Overall, for this scenario, either method works.

# ---
# ## Part 5 – When $k$-Means Fails

# ### 5a — Non-Spherical Clusters

# In[109]:


# Load two-moons dataset
#  load data/p3b_two_moons.csv
file_str_p5a = "p3b_two_moons.csv"
df_p5a = pd.read_csv("{}\\{}".format(directory_str,file_str_p5a))
print(df_p5a.head(10))

df_p5a_summary_stats = df_p5a.describe()
c_names_df_p5a = df_p5a.columns
numr_df_p5a = df_p5a[c_names_df_p5a[0]].value_counts()
print(df_p5a_summary_stats)

X = df_p5a[c_names_df_p5a[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p5a[c_names_df_p5a[-1]]  # label column

p5a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p5a.custom_X(X)
p5a.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":2})

# Plot with true cluster colors
#  scatter plot, color by true_cluster
p5a.calculate_label_bmask_clustering(p5a.y)
p5a.plot_clustering(plt,p5a.X,0,1,label_colors=["red","blue","green","orange"],label_labels=["True Label 1","True Label 2"])
p5a.plot_labels_show("x1","x2","x2 vs. x1")


# In[110]:


# K-means with k=2
#  KMeans(n_clusters=2, random_state=42)
# km_p5a = None

# Report silhouette score
#  print silhouette score
p5a.key_metrics_clustering()
p5a.print_key_metrics_clustering()


# In[111]:


# Side-by-side comparison: true clusters vs k-means assignment
#  1x2 subplot showing the discrepancy
p5a.calculate_label_bmask_clustering(p5a.y)
p5a.plot_clustering(plt,p5a.X,0,1,label_colors=["red","blue"],s=200,label_labels=["True Label 1","True Label 2"])
p5a.calculate_label_bmask_clustering(p5a.labels)
p5a.plot_clustering(plt,p5a.X,0,1,label_colors=["green","orange"],s=75,label_labels=["Predicted Label 1","Predicted Label 2"])
p5a.plot_labels_show("x1","x2","x2 vs. x1")

fig, (ax1, ax2) = plt.subplots(1, 2)

p5a.calculate_label_bmask_clustering(p5a.y)
p5a.plot_clustering(ax1,p5a.X,0,1,label_colors=["purple","yellow","gray"])
p5a.plot_clusters_same(ax1,0,1)
p5a.calculate_label_bmask_clustering(p5a.labels)
p5a.plot_clustering(ax2,p5a.X,0,1,label_colors=["purple","yellow","gray"])
p5a.plot_clusters_same(ax2,0,1)
p5a.plot_labels_show_sub(fig,"x1","x2","x2 vs. x1")


# **5a Analysis:**
# 
# **Geometric assumption violated:** 
# 
# The K-means algorithm assumes spherical clustering, not the half-moon data in this problem.
# 
# **Why k-means splits each moon:** 
# 
# K-means minimizes the Within-Cluster Sum of Squares (WCSS) of the data. This essentially leads to K-means creating two centroids that each consist mostly of one half-moon and partially of the other half-moon. This is the minimum distance squared solution created by the WCSS equation.
# 
# **Alternative algorithm and its principle:** 
# 
# An alternative algorithm would be DBSCAN which looks at density and can identify more complex shapes such as the half-moon in this problem. Another approach is hierarchical clustering which creates a tree-hierarchy of the data. This algorithm could also solve the problem but is much more computationally expensive.

# ### 5b — Clusters with Different Densities

# In[112]:


# Load density dataset
#  load data/p3c_densities.csv
file_str_p5b = "p3c_densities.csv"
df_p5b = pd.read_csv("{}\\{}".format(directory_str,file_str_p5b))
print(df_p5b.head(10))

df_p5b_summary_stats = df_p5b.describe()
c_names_df_p5b = df_p5b.columns
numr_df_p5b = df_p5b[c_names_df_p5b[0]].value_counts()
print(df_p5b_summary_stats)

X = df_p5b[c_names_df_p5b[:2]]
y = df_p5b[c_names_df_p5b[-1]]

p5b = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p5b.custom_X(X)
p5b.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":3})

# Plot with true cluster colors
#  scatter plot, color by true_cluster
p5b.calculate_label_bmask_clustering(p5b.y)
p5b.plot_clustering(plt,p5b.X,0,1,label_colors=["red","blue","green","orange"],label_labels=["True Label 1","True Label 2","True Label 3"])
p5b.plot_labels_show("x1","x2","x2 vs. x1")

# Plot with true cluster colors
#  scatter plot colored by true_cluster


# In[113]:


# Apply k-means with k=3
#  KMeans(n_clusters=3, random_state=42), compare with true labels
#  side-by-side plot: true vs k-means
p5b.calculate_label_bmask_clustering(p5b.y)
p5b.plot_clustering(plt,p5b.X,0,1,label_colors=["red","blue","green"],s=200,label_labels=["True Label 1","True Label 2","True Label 3"])
p5b.calculate_label_bmask_clustering(p5b.labels)
p5b.plot_clustering(plt,p5b.X,0,1,label_colors=["orange","purple","pink"],s=75,label_labels=["Predicted Label 1","Predicted Label 2","Predicted Label 3"])
p5b.plot_labels_show("x1","x2","x2 vs. x1")

fig, (ax1, ax2) = plt.subplots(1, 2)

p5b.calculate_label_bmask_clustering(p5b.y)
p5b.plot_clustering(ax1,p5b.X,0,1,label_colors=["purple","yellow","gray"])
p5b.plot_clusters_same(ax1,0,1)
p5b.calculate_label_bmask_clustering(p5b.labels)
p5b.plot_clustering(ax2,p5b.X,0,1,label_colors=["purple","yellow","gray"])
p5b.plot_clusters_same(ax2,0,1)
p5a.plot_labels_show_sub(fig,"x1","x2","x2 vs. x1")


# In[114]:


# ARI between k-means and true labels
#  print adjusted_rand_score

p5b.key_metrics_clustering()
p5b.print_key_metrics_clustering()


# **5b Analysis:**
# 
# **Why k-means misplaces the boundary:** 
# 
# K-means misplaces the boundary because Label 3 has a lot of variance and thus a sizeable portion of its points are too close to the other clusters.
# 
# **Model-based approach that relaxes variance assumption:** 
# 
# A model that would take the variance of each cluster into account would be the Gaussian Mixture Models, which are more computationally expensive but would mostly be able to account for the variance shown in this dataset.

# ### 5c — Initialization Sensitivity

# In[115]:


# Load initialization dataset
#  load data/p3d_init.csv
file_str_p5c = "p3d_init.csv"
df_p5c = pd.read_csv("{}\\{}".format(directory_str,file_str_p5c))
print(df_p5c.head(10))

df_p5c_summary_stats = df_p5c.describe()
c_names_df_p5c = df_p5c.columns
numr_df_p5c = df_p5c[c_names_df_p5c[0]].value_counts()
print(df_p5c_summary_stats)

X = df_p5c[c_names_df_p5c[:2]]  # features: vibration_rms, exhaust_temp_deviation
y = df_p5c[c_names_df_p5c[-1]]  # label column

# Run k-means 50 times with random init, seeds 0–49
wcss_runs_2 = []
wcss_runs_1000 = []
wcss_runs_k__ = []
for seed in range(50):
    p5c = ModelML("K-Clustering",X,y,random_state = seed) # "LogisticRegression",
    p5c.custom_X(X)
    p5c.full_model_process_clustering(kwargs_model={"max_iter":2,"n_clusters":3,"n_init":1,"init":"random"})
    wcss_runs_2.append(p5c.WCSS)
    p5c.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":3,"n_init":1,"init":"random"})
    wcss_runs_1000.append(p5c.WCSS)
    p5c.full_model_process_clustering(kwargs_model={"max_iter":2,"n_clusters":3,"init":'k-means++',"n_init":1})
    wcss_runs_k__.append(p5c.WCSS)
    # pass  #  KMeans(n_clusters=3, init='random', n_init=1, random_state=seed)
    #       #       append inertia_ to wcss_runs

p5c = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p5c.custom_X(X)
p5c.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":3})

p5c.calculate_label_bmask_clustering(p5c.y)
p5c.plot_clustering(plt,p5c.X,0,1,label_colors=["red","blue","green","orange"],label_labels=["True Label 1","True Label 2"])
p5c.plot_labels_show("x1","x2","x2 vs. x1")


# In[116]:


# Histogram of WCSS values; mark mean and minimum
#  plt.hist(wcss_runs, bins=15); mark mean and min with vertical lines
# print(wcss_runs_2)

plot_min = min([min(wcss_runs_2),min(wcss_runs_1000)])
plot_max = max([max(wcss_runs_2),max(wcss_runs_1000)])
plt.hist(wcss_runs_1000, bins=25, range=(plot_min,plot_max),label="iterations = 1000"); # mark mean and min with vertical lines
plt.hist(wcss_runs_2, bins=25, range=(plot_min,plot_max),label="iterations = 2"); # mark mean and min with vertical lines
minimum,mean_val = [min(wcss_runs_2),np.mean(wcss_runs_2)]
plt.plot([minimum,minimum],[0,seed],label="minimum iter=2"); # mark mean and min with vertical lines
plt.plot([mean_val,mean_val],[0,seed],label="mean iter=2"); # mark mean and min with vertical lines
plt.legend()
p5c.plot_labels_show("x1","x2","x2 vs. x1")


# In[117]:


# k-means++ for comparison
#  KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=42)
#  print its inertia_ and show where it falls in the histogram
# p5c = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
# p5c.custom_X(X)
# p5c.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":3,"init":'k-means++',"n_init":1})


# In[118]:


print(wcss_runs_2)
print(wcss_runs_1000)
print(wcss_runs_k__)
plot_min = min([min(wcss_runs_2),min(wcss_runs_1000),min(wcss_runs_k__)])
plot_max = max([max(wcss_runs_2),max(wcss_runs_1000),max(wcss_runs_k__)])
plt.hist(wcss_runs_k__, bins=25, range=(plot_min,plot_max),label="k++ iterations = 2"); # mark mean and min with vertical lines
plt.hist(wcss_runs_1000, bins=25, range=(plot_min,plot_max),label="iterations = 1000"); # mark mean and min with vertical lines
plt.hist(wcss_runs_2, bins=25, range=(plot_min,plot_max),label="iterations = 2"); # mark mean and min with vertical lines
minimum,mean_val = [min(wcss_runs_2),np.mean(wcss_runs_2)]
plt.plot([minimum,minimum],[0,seed],label="minimum iter=2"); # mark mean and min with vertical lines
plt.plot([mean_val,mean_val],[0,seed],label="mean iter=2"); # mark mean and min with vertical lines
minimum,mean_val = [min(wcss_runs_k__),np.mean(wcss_runs_k__)]
plt.plot([minimum,minimum],[0,seed],label="k++ minimum iter=2"); # mark mean and min with vertical lines
plt.plot([mean_val,mean_val],[0,seed],label="k++ mean iter=2"); # mark mean and min with vertical lines
plt.legend()
p5c.plot_labels_show("x1","x2","x2 vs. x1")


# **5c Analysis:**
# 
# **Multiple WCSS values and the optimization landscape:** 
# 
# K-means is randomly initiated from a point within the dataset. This means that if there are too few iterations and the initial point is a bad one, then the algorithm may select a local minimum instead of a global minimum.
# 
# **How k-means++ improves initialization (3–5 sentences):** 
# 
# The k-means++ algorithm massively reduces the number of iterations to find a minimum. It does this by using a probability distribution method instead of a random point of initialization. After the first random center the next are chosen using a probability scaling proportionally to the nearest existing center. In each iteration only one potential location for moving a single cluster center is sampled. Scikit uses Greedy K-means++, which samples multiple cluster centers each iteration and chooses the best. This style of less random initialization is viewed as a somewhat reliable way to improve K-means convergence rates.
# 
# **What n_init does and recommended value:** 
# 
# N_init describes the number of times the centroid algorithm is run with different seeds and the final output is the result with the best inertia. By default with init=”random” the value is 10 and with K-means++ the value is 1 because K-means++ plus is slightly more advanced algorithm as already explained.

# ---
# ## Part 6 – Integration: Classification or Clustering?

# In[119]:


# Load integration dataset (no labels)
#  load data/p4_integration.csv into df_p6
file_str_p6a = "p4_integration_labels.csv"
df_p6a = pd.read_csv("{}\\{}".format(directory_str,file_str_p6a))
print(df_p6a.head(10))

df_p6a_summary_stats = df_p6a.describe()
c_names_df_p6a = df_p6a.columns
numr_df_p6a = df_p6a[c_names_df_p6a[0]].value_counts()
print(df_p6a_summary_stats)

X = df_p6a[c_names_df_p6a[:-1]]
y = df_p6a[c_names_df_p6a[-1]]

p6a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p6a.custom_X(X)
p6a.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":3})

p6a.calculate_label_bmask_clustering(p6a.y)
p6a.plot_clustering(plt,p6a.X,0,1,label_colors=["red","blue","green","orange"],label_labels=["True Label 1","True Label 2","True Label 3"])
p6a.plot_clusters_same(plt,0,1)
p6a.plot_labels_show("x1","x2","x2 vs. x1")

p6a.confusion_matrix_calculation()
p6a.confusion_matrix
# Display summary statistics
#  df_p6.describe()


# **Decision — Classification or Clustering?**
# 
# [Your answer here: justify your choice based on what information is available.]
# 
# Clustering has to be used because the data has no labels. If it did have labels then logistic regression would likely yield better insights.

# In[120]:


# PCA: reduce to 2 principal components
#  StandardScaler -> PCA(n_components=2) -> scatter in PC space
p6a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p6a.custom_X(X)
p6a.full_model_process_clustering(pca="yes",kwargs_model={"max_iter":1000,"n_clusters":3},kwargs_pca={"n_components":2})


# In[121]:


#  scatter in PCA space colored by k-means labels
p6a.calculate_label_bmask_clustering(p6a.y)
# print(p6a.X_main)
p6a.plot_clustering(plt,p6a.X_main,0,1,label_colors=["red","blue","green","orange"],label_labels=["True Label 1","True Label 2","True Label 3"])
p6a.plot_clusters_same(plt,0,1)
p6a.plot_labels_show("x1","x2","x2 vs. x1")


# In[122]:


# Choose k using elbow and silhouette, then apply k-means
#  elbow and silhouette analysis on the original (scaled) 6D data
k_range = range(1, 11)
wcss,sil_scores,ARI_score = [[],[],[]]
wcss_pca,sil_scores_pca,ARI_score_pca = [[],[],[]]

for k in k_range:
    p6a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
    p6a.custom_X(X)
    p6a.full_model_process_clustering(kwargs_model={"max_iter":1000,"n_clusters":k})
    if k != 1:
        p6a.key_metrics_clustering()
        sil_scores.append(p6a.silhouette_score)
    else:
        p6a.ARI = adjusted_rand_score(p6a.y,p6a.labels[0])
        sil_scores.append(0)
    wcss.append(p6a.WCSS)
    ARI_score.append(p6a.ARI)

    p6a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
    p6a.custom_X(X)
    p6a.full_model_process_clustering(pca="yes",kwargs_model={"max_iter":1000,"n_clusters":k},kwargs_pca={"n_components":2})
    if k != 1:
        p6a.key_metrics_clustering()
        sil_scores_pca.append(p6a.silhouette_score)
    else:
        p6a.ARI = adjusted_rand_score(p6a.y,p6a.labels[0])
        sil_scores_pca.append(0)
    wcss_pca.append(p6a.WCSS)
    ARI_score_pca.append(p6a.ARI)
print(sil_scores)
print(wcss)
print(sil_scores_pca)
print(wcss_pca)
#  fit final k-means, report silhouette score
plt.plot(k_range,wcss,color="red",label="Scaled")
plt.plot(k_range,wcss_pca,color="blue",label="Scaled w/ PCA")
plt.scatter(k_range[2],wcss[2],label="Elbow Point",color="green")
plt.scatter(k_range[2],wcss_pca[2],label="Elbow Point",color="orange")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.title("WCSS vs. k")
plt.legend()
plt.show()

plt.plot(k_range,sil_scores,color="red",label="Scaled")
plt.plot(k_range,sil_scores_pca,color="blue",label="Scaled w/ PCA")
plt.scatter(k_range[2],sil_scores[2],label="Elbow Point",color="green")
plt.scatter(k_range[2],sil_scores_pca[2],label="Elbow Point",color="orange")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. k")
plt.legend()
plt.show()


# In[123]:


# Reveal true labels and compute ARI
#  load data/p4_integration_labels.csv
#  adjusted_rand_score(true_health_state, km_labels)
#  print and side-by-side scatter comparing k-means vs true labels in PCA space
print(ARI_score)
print(ARI_score_pca)
plt.plot(k_range,ARI_score,color="red",label="Scaled")
plt.plot(k_range,ARI_score_pca,color="blue",label="Scaled w/ PCA")
plt.scatter(k_range[2],ARI_score[2],label="Elbow Point",color="green")
plt.scatter(k_range[2],ARI_score_pca[2],label="Elbow Point",color="orange")
plt.xlabel("k")
plt.ylabel("ARI Score")
plt.title("ARI Score vs. k")
plt.legend()
plt.show()

# p6a.calculate_label_bmask_clustering(p6a.y)
# print(p6a.X_main)
p6a = ModelML("K-Clustering",X,y,random_state = 42) # "LogisticRegression",
p6a.custom_X(X)
p6a.full_model_process_clustering(pca="yes",kwargs_model={"max_iter":1000,"n_clusters":3},kwargs_pca={"n_components":2})

p6a.calculate_label_bmask_clustering(p6a.y)
p6a.plot_clustering(plt,p6a.X_main,0,1,label_colors=["red","blue","green",],s=150,label_labels=["True Label 1","True Label 2","True Label 3"])
p6a.calculate_label_bmask_clustering(p6a.labels)
p6a.plot_clustering(plt,p6a.X_main,0,1,label_colors=["orange","purple","yellow"],label_labels=["Predicted Label 1","Predicted Label 2","Predicted Label 3"])
p6a.plot_clusters_same(plt,0,1)
p6a.plot_labels_show("x1","x2","x2 vs. x1")

p6a.confusion_matrix_calculation()
p6a.confusion_matrix


# **Part 6 Analysis:**
# 
# **Did k-means recover health states?:** 
# 
# The k-means did recover the health states but with slightly less accuracy than with a full rank model. There were two of either a false negative or false positive in the confusion matrix of the PCA reduced model.
# 
# **How would the analysis change with labels from the start?:** 
# 
# It was incredibly easy to tell how many clusters appeared in the data even without labels. There were three clear clusters that appeared in the first two columns of the data. However, since the model has six columns, it would normally be extremely difficult to determine the number of clusters by inspection. Having the labels from the start would help inform the user about any potential outliers and whether K-means would be a useful algorithm or whether another clustering algorithm would be better.
# 
# **Additional information needed for a real deployment decision:** 
# 
# It would probably also be good to use the logistic regression algorithm to quantify exact probabilities and fine tune probability cutoffs. K-means only looks at the feature space and can possibly miss patterns that other algorithms may find. However, K-means was a good first pass to identify whether the data was able to be classified. 
