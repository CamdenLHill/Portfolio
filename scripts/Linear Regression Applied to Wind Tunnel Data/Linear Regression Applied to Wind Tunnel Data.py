#!/usr/bin/env python
# coding: utf-8

# ## Name: Camden Hill
# ## Date: 2/16/2026
# 
# ---

# ## Package Imports
# 
# Import all required packages in this cell. The notebook should run completely from this point forward.

# In[ ]:


# Data manipulation
import numpy as np
import pandas as pd
import os
import statistics as st

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,max_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from statsmodels.tools.tools import add_constant
from sklearn.utils import shuffle

# Statistical analysis (for Part 2)
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# get_ipython().run_line_magic('matplotlib', 'inline')

# import sys
# from io import StringIO


# Ensure reproducibility
np.random.seed(42)

print("All packages imported successfully!")


# ---
# # Part 1: Applied Linear Regression

# ## Part 1a:
# 
# **Tasks:**
# 1. Load the drag coefficient data from `data/drag_data.csv`
# 2. Display the first 10 rows
# 3. Compute summary statistics
# 4. Discuss feature scales and implications

# In[ ]:


# Load the data
#  Load data from data/drag_data.csv

# Loads and displays data
current_dir = os.getcwd()
file_path_rel = "drag_data.csv"
full_path = os.path.join(current_dir, file_path_rel)
full_path = # \drag_data.csv"
# print(full_path)
df = pd.read_csv(full_path)


# In[ ]:


# Display first 10 rows
#  Show first 10 rows of the dataset
df.head(10)


# In[ ]:


# Summary statistics
#  Compute and display summary statistics for all variables
c_names = df.columns
num_columns = len(c_names)
summary_statistics = {}
print(df[c_names[0]])
for i in range(num_columns):
    av = np.average(df[c_names[i]])
    stdp = np.std(df[c_names[i]])
    stdp_1 = stats.tstd(df[c_names[i]])
    stdp_2 = st.stdev(df[c_names[i]])
    # ftest = 
    val_min = min(df[c_names[i]])
    val_max = max(df[c_names[i]])
    summary_statistics[c_names[i]] = {"average":av, "standard_deviation_st":stdp_2, "standard_deviation_np": stdp, "standard_deviation_spy": stdp_1, "minimum":val_min, "maximum":val_max}
    # print(summary_statistics[c_names[i]])
    print("{}: average = {}, standard deviation sample = {}, minimum = {}, maximum = {}".format(c_names[i],summary_statistics[c_names[i]]["average"],summary_statistics[c_names[i]]["standard_deviation_st"],summary_statistics[c_names[i]]["minimum"],summary_statistics[c_names[i]]["maximum"],))


# ### Analysis: Feature Scales
# 
# ** Write 1-2 paragraphs discussing:**
# - Which features span multiple orders of magnitude?
# - Why might this be problematic for linear regression?
# - What strategies could address this issue?
# 
# The Reynold's Number is the only variable that could be said to span multiple orders of magnitude, as the standard deviation is roughly seven orders of magnitude. The variable most comparable in variance to the Reynolds Number is the angle of attack, but even that only spans roughly a single order of magnitude.
# 
# There are multiple reasons why large differences in the order of magnitude of model variables cause issues, such as issues with gradient descent, regularization, and numerical stability. However, the act of standardizing fixes all of these issues by essentially reducing each variable into a z-score. So, the average and standard deviation of a column of data is found, and then each individual point in that column is standardized into a z-score relative to its column's summary statistics. This allows regression models to examine a data set from a statistically standardized perspective. Without data standardization, regression models such as Lasso or Ridge will penalize larger features unfairly, and Gradient Descent will be less efficient because it will be harder to minimize the cost function. Additionally, computers always have floating-point error, so larger numbers will gradually result in increasing precision errors the more calculation is done.
# 

# ## Part 1b: Data Visualization and Regime Identification
# 
# **Tasks:**
# 1. Create scatter plots: C_D vs each feature
# 2. Distinguish subsonic/transonic regimes in C_D vs Mach plot
# 3. Create correlation matrix heatmap
# 4. Analyze relationships and patterns

# In[ ]:


# Scatter plots: C_D vs features
#  Create scatter plots showing C_D vs alpha, Mach, Reynolds
# For C_D vs Mach, color-code by regime (subsonic M < 0.7, transonic M >= 0.7)
plt.scatter(df[c_names[0]],df[c_names[-1]],color=(np.random.random(), np.random.random(), np.random.random()))
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.show()


# In[ ]:


plt.scatter(df[c_names[1]],df[c_names[-1]],label="C_D vs. Mach",color=(np.random.random(), np.random.random(), np.random.random()))
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()


# In[ ]:


plt.scatter(df[c_names[2]],df[c_names[-1]],color=(np.random.random(), np.random.random(), np.random.random()))
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.show()


# In[ ]:


# Correlation matrix
#  Create and display correlation matrix heatmap
co_mtx = df.corr(numeric_only=True)
print(co_mtx)
sns.heatmap(co_mtx,cmap="YlGnBu", annot=True)
plt.show()


# ### Analysis: Relationships and Patterns
# 
# ** Write 1-2 paragraphs discussing:**
# - Which features show strong linear relationships with C_D?
# - Different behavior in subsonic vs transonic regimes?
# - Any obvious outliers or unusual patterns?
# 
# The Mach Number and Reynolds Number both show strong linear relationships with the C_D, as shown by the numbers in the bottom row of the above correlation matrix. The Mach Number has twice the linear correlation value as the Reynolds Number and is positive, while the Reynolds Number has a negative correlation. The two graphs above the correlation matrix, C_D vs. Mach and C_D vs. Reynolds, visually show this relationship. However, the Mach Number and C_D relationship is clearly non-linear and looks, from visual inspection, to possibly be quadratic. There are several points that appear to be outliers on their respective graphs.
# 
# It's difficult to tell from the graph, but the Mach Number and C_D relationship appears to possibly exhibit different characteristics between the subsonic and transonic regions. However, from visual observation, the relationship could just be quadratic. Additionally, there appears to be nearly as much correlation between alpha and the Reynolds Number as there is between alpha and the C_D. Meaning that in the future formulation of the model, the relationship between alpha and Reynolds Number should be explored. Several points around the transonic regime appear to have extremely high coefficients of drag. While the graph of C_D vs. alpha and C_D vs. Reynolds Number have outliers but nothing too extreme.

# ## Part 1c: Feature Engineering
# 
# **Tasks:**
# 1. Create Feature Set A: General polynomial features (degree 2) with StandardScaler
# 2. Create Feature Set B: Physics-inspired features based on aerodynamic theory
# 3. Document and justify your choices

# In[ ]:


# Prepare features and target
#  Extract features (alpha, Mach, Reynolds) and target (C_D)
X = df[c_names[:-1]]
y = df[c_names[-1]]
print(X)
print(y)


# In[ ]:


# Feature Set A: General Polynomial Features
#  Create polynomial features up to degree 2
general_poly = PolynomialFeatures(degree = 2, include_bias = False) # , include_bias = True
X_gen = general_poly.fit_transform(X)
print(X_gen)
X_traing, X_testg, y_traing, y_testg = train_test_split(X_gen, y, test_size=0.2, random_state=42)
# X_gen = general_poly.fit_transform(X)

#  Apply StandardScaler
scaler_gen = StandardScaler()
X_gen_scaled_train = scaler_gen.fit_transform(X_traing)
X_gen_scaled_test = scaler_gen.transform(X_testg)
mean_gen = scaler_gen.mean_
SD_gen = scaler_gen.scale_
print("General Polynomial Scaled Fit Matrix:")
print(X_gen_scaled_train)


# In[ ]:


# Feature Set B: Physics-Inspired Features
#  Engineer features based on aerodynamic theory
# Hints:
# - Induced drag: related to alpha^2
# - Reynolds drag: consider Re^(-0.2) or log(Re) for multi-order variation
# - Compressibility: Prandtl-Glauert correction ~ 1/sqrt(1-M^2)
# - Wave drag: emerges in transonic, consider (M - M_crit)^2 for M > M_crit
# - Interactions: alpha*M, etc.

num_rows = len(df[c_names[0]])

# # # 1, a, M, Re, a^2, a x M, a x Re, M^2, M x Re, Re^2, log(Re), 1/sqrt(1-M^2), (M - M_crit)^2
# X_phys = np.zeros((num_rows, 13))
# # print(X_phys[:,1:4])
# X_phys_sub[:,0] = [1 for i in range(num_rows)] # 1
# # X_phys_sub[:,1] = df[c_names[0]] # a
# X_phys[:,2] = df[c_names[1]] # M
# # X_phys[:,3] = df[c_names[2]] # Re
# # X_phys[:,4] = df[c_names[0]]**2 # a^2
# # X_phys[:,5] = df[c_names[0]]*df[c_names[1]] # a x M
# # X_phys[:,6] = df[c_names[0]]*df[c_names[2]] # a x Re
# # X_phys[:,8] = df[c_names[0]]*df[c_names[2]] # a x Re
# X_phys[:,7] = df[c_names[1]]**2 # M^2
# # X_phys[:,7] =  np.log(df[c_names[1]]) # M^2
# # X_phys[:,8] = df[c_names[1]]**3 # M^3
# # X_phys[:,8] = df[c_names[1]]*df[c_names[2]] # M x Re
# # X_phys[:,8] = df[c_names[2]]**2 # Re^2
# # X_phys[:,9] = df[c_names[2]]**(-0.2) # log(Re)
# # X_phys[:,10] = np.log10(df[c_names[2]]) # log(Re)
# X_phys[:,10] = np.log(df[c_names[2]]) # log(Re)
# # X_phys[:,9] = np.log(df[c_names[2]]) # log(Re)
# X_phys[:,11] = 1/((1 - (df[c_names[1]]**2))**0.5) # 1/sqrt(1-M^2)
# # for i in range(num_rows):
# #     if df[c_names[1]][i] > 0.7:
# #         X_phys[i,11] = (df[c_names[1]][i] - 0.7)**2 # (M - M_crit)^2
# #     else:
# #         X_phys[i,11] = 1/((1 - (df[c_names[1]][i]**2))**0.5)
# for i in range(num_rows):
#     if df[c_names[1]][i] > 0.7:
#         X_phys[i,12] = (df[c_names[1]][i] - 0.7)**2 # (M - M_crit)^2
#     else:
#         X_phys[i,12] = 0

X_phys = np.zeros((num_rows, 6))
# X_phys[:,0] = [1 for i in range(num_rows)] # 1
# if df[c_names[1]][i] > 0.7:
#     X_phys[i,1] = (df[c_names[1]][i] - 0.7)**2 # (M - M_crit)^2
# else:
#     X_phys[i,1] = 0
X_phys[:,0] = df[c_names[0]]*df[c_names[1]] # a x M
X_phys[:,1] = df[c_names[0]]*df[c_names[2]] # a x Re
X_phys[:,2] = df[c_names[1]] # M
# X_phys[:,2] = np.log(df[c_names[1]]) # log(M^2)
X_phys[:,3] = df[c_names[1]]**2 # M^2
X_phys[:,4] = np.log(df[c_names[2]]) # log(Re)
# X_phys[:,4] = df[c_names[2]]**(-0.2) # log(Re)
X_phys[:,5] = 1/((1 - (df[c_names[1]]**2))**0.5) # 1/sqrt(1-M^2)

# print(df[c_names[1]][0]*df[c_names[2]][0])
print("First Row of Physics-Based Fit Matrix: ", X_phys[0,:])
print("Last Row of Physics-Based Fit Matrix: ", X_phys[-1,:])
#  Apply StandardScaler
indices_phys = np.arange(len(X_phys[:,0]))
X_trainp, X_testp, y_trainp, y_testp, train_indicesp, test_indicesp = train_test_split(X_phys, y, indices_phys, test_size=0.2, random_state=42)

scaler_phys = StandardScaler()
X_phys_scaled_train = scaler_phys.fit_transform(X_trainp)
X_phys_scaled_test = scaler_phys.transform(X_testp)
mean_phys = scaler_phys.mean_
SD_phys = scaler_phys.scale_


# ### Justification: Physics-Based Features
# 
# ** Write 1-2 paragraphs explaining:**
# - Why you chose specific physics-based features
# - How they relate to the drag decomposition (induced, viscous, wave, compressibility)
# - How you handled the multi-order magnitude Reynolds number variation
# 
# Multiple different combinations of variables were tested initially. I assumed that if I added some physics-based features on top of the already existing polynomial variables, it would improve the regression. However, this wasn't the case. This actually decreased the overall regression. The next thing I tried was to vary whether the model contained terms for dynamic pressure, Prandtl-Glauert correction, induced drag, Reynolds drag, and wave drag instead of containing all of them. This process took a while but ultimately resulted in a model containing 7-terms. The first term was to account for the intercept, which, in a real-world situation, would be zero if the data spanned all the way to a Mach Number of zero. The next term was a correlation term between alpha and Reynolds number. For some reason, this factor consistently improved the regression of the data when a variety of other factors were unable to. The third term is another correlation term between alpha and Mach Number, which didn't seem to affect the regression rate that much, but makes physical sense to include, as increasing alpha increases the local Mach Number in certain regions. The fourth term is the Mach Number, which although it isn’t directly associated with an equation it doesn’t hurt the regression model and provides somewhat better results. The fifth term is the Mach Number squared as this is directly related to the dynamic pressure, which affects a variety of real-world equations. The sixth term is the logarithm of the Reynolds Number. This sixth term consistently improved regression and seemed to be superior in effect to the Reynolds Number to the negative 0.2 power. The final term is then compressibility, which consistently improved regression by a small amount. This 7-term model clearly leaves out some known physical effects; however, various combinations of physical effects were tested and compared to see if they improved regression to reduce the number of terms to just 7. So, for example, induced drag which relates to alpha squared seemed to reduce the regression of the data. Therefore, it didn’t make sense to include it in the final model. The multi-order Reynolds Magnitude was handled through scaling and by using a logarithm fit.

# ## Part 1d: Model Training and Evaluation
# 
# **Tasks:**
# 1. Split data: 80% train, 20% test (fixed random_state=42)
# 2. Train Model A (polynomial features) and Model B (physics-inspired features)
# 3. Compute metrics: R², Adjusted R², RMSE, MAE (train and test)
# 4. Compare model performance

# In[ ]:


# Train-test split
#  Split both feature sets and target into train/test (80/20, random_state=42)

print("Splitting was done previously for both the physics-based and general polynomial models because fitting the scaled X before splitting it")
print("allows data leakage. Meaning the test data is included in the training of the model. Which, invalidates the point of splitting the")
print("data at all. Results are similar between leakage and no leakage but I had to go back once I had already completed the majority of")
print("the assignment to make this change. Therefore, a lot of this assignment has had to be recoded and rewritten.")

# # 1. Load General Polynomial and split data
# X_traing, X_testg, y_traing, y_testg = train_test_split(
# X_gen_scaled, y, test_size=0.2, random_state=42
# )

# # 1. Load Physics-Based and split data
# X_trainp, X_testp, y_trainp, y_testp = train_test_split(
# X_phys_scaled, y, test_size=0.2, random_state=42
# )


# In[ ]:


# Model A: Polynomial Features
#  Train LinearRegression model on polynomial features
# 2. Create General Polynomial and train model
modelg = LinearRegression()
modelg.fit(X_gen_scaled_train, y_traing)

# 2. Create Physics-Based and train model
modelp = LinearRegression()
modelp.fit(X_phys_scaled_train, y_trainp)
#  Compute predictions on train and test sets
# 3. Make General Polynomial predictions
y_predg = modelg.predict(X_gen_scaled_test)

# 3. Make Physics-Based predictions
y_predp = modelp.predict(X_phys_scaled_test)


# In[ ]:


# Compute metrics for both models
#  Calculate R², Adjusted R², RMSE, MAE for train and test sets
# 4. General Polynomial Evaluate
interceptg = modelg.intercept_
coefficientsg = modelg.coef_
coefficients_numg = len(coefficientsg)

n_gen = len(y_testg)
mseg = mean_squared_error(y_testg, y_predg)
rmseg = np.sqrt(mseg)
r2g = r2_score(y_testg, y_predg)
r2ag = 1 - ((1-r2g)*(n_gen-1)/(n_gen - (coefficients_numg + 1) - 1)) # plus 1 for d to account for intercept
maeg = mean_absolute_error(y_testg, y_predg)
mapeg = mean_absolute_percentage_error(y_testg, y_predg)
maxeg = max_error(y_testg, y_predg)
# cvsg = cross_val_score(LinearRegression(),X_gen_scaled,y)

# 4. Physics-Based Evaluate
interceptp = modelp.intercept_
coefficientsp = modelp.coef_
coefficients_nump = len(coefficientsp)

n_phys = len(y_testp)
msep = mean_squared_error(y_testp, y_predp)
rmsep = np.sqrt(msep)
r2p = r2_score(y_testp, y_predp)
r2ap = 1 - ((1-r2p)*(n_phys-1)/(n_phys - (coefficients_nump + 1) - 1)) # plus 1 for d to account for intercept
maep = mean_absolute_error(y_testp, y_predp)
mapep = mean_absolute_percentage_error(y_testp, y_predp)
maxep = max_error(y_testp, y_predp)
# cvsp = cross_val_score(LinearRegression(),X_phys_scaled,y)
# Remember: Adjusted R² = 1 - (1-R²)*(n-1)/(n-p-1)


# In[ ]:


# Create comparison table
#  Display metrics in a comparison table
reg_metrics = {"GP":{"MSE":mseg,"RMSE":rmseg,"R^2":r2g,"R_adj^2":r2ag,"MAE":maeg,"MAPE":mapeg,"Max Error":maxeg},#,"CV Score":cvsg},
               "PB":{"MSE":msep,"RMSE":rmsep,"R^2":r2p,"R_adj^2":r2ap,"MAE":maep,"MAPE":mapep,"Max Error":maxep}}#,"CV Score":cvsp}}

print("Key Metrics: MSE,                  RMSE,                  R^2,                R_adj^2,            MAE,                   MAPE,                 Max Error")
print("GP:          {}, {}, {}, {}, {}, {}, {}".format(mseg,rmseg,r2g,r2ag,maeg,mapeg,maxeg))
print("PB:          {}, {}, {}, {}, {}, {}, {}".format(msep,rmsep,r2p,r2ap,maep,mapep,maxep))

print("r^2, Polynomial 10 parameter = 0.8432590063877778")
print("r^2, original 8 parameter = 0.8474602589090668")
print("r^2, original 13 parameter = 0.83757028238973")
print("r^2, with piecewise combined prandtl & wave * log(Re) = 0.8279396175442874")
print("r^2, with piecewise combined prandtl & wave * Re^(-0.2) = 0.8279396175442875")
print("r^2, Re^(-0.2) = 0.834979897723414")
print("r^2, original 8 parameter w/ piecewise combined prandtl & wave = ~0.82")
print("r^2, original 8 parameter w/ Re^(-02) = 0.8248839547435652")
print("r^2, original 8 parameter w/ correlated = 0.826823558804907")
print("r^2, original 8 parameter w/ correlated axRe = 0.8407022573181913")
print("r^2, original 8 parameter w/ M^2 = 0.8468706446463296")
print("r^2, original 8 parameter w/ log(M) = 0.8465918932055368")
print("r^2, original 8 parameter w/ M^2 & axRe = 0.8539679560254726")
print("r^2, original 8 parameter w/ M^2 & axRe & Mcrit = 0.72 = 0.8550378538526808")
print("r^2, original 8 parameter w/ M^2 & axRe & Mcrit = 0.75 = 0.8568517084758079")
print("r^2, original 8 parameter w/ M^2 & axRe & Mcrit = 0.7525 = 0.8569631423869367")
print("After-testing Physics-Based Form:")
print("1, a, M, Re, a x Re, log(Re), 1/sqrt(1-M^2), (M - M_crit)^2 @ Mcrit > 0.7525")
print("1, a x Re, log(Re), M^2, 1/sqrt(1-M^2), (M - M_crit)^2 @ Mcrit > 0.7 = 0.8800624359436029, but C_D decreases with M^2, which isn't physically accurate.")
print("Final 7-term Physics-Based Form:")
# print("1, a x Re, log(Re), M^2, 1/sqrt(1-M^2), (M - M_crit)^2 @ Mcrit > 0.7 = 0.8792451128665608")
print("1, a x M, a x Re, M, M^2, log(Re), 1/sqrt(1-M^2) = 0.8794251462926067")




# ### Analysis: Model Comparison
# 
# ** Write 2-3 paragraphs discussing:**
# - Which model performs better on test data?
# - Is there evidence of overfitting? (compare train vs test)
# - Which metrics are most meaningful for this application?
# 
# The physics-based model initially performed worse than the polynomial fit, but after comparing the effects of different regression equations on the model performance, the physics-based model performed better. However, the question is whether the model overfits the data, because if it does, it would invalidate the physical model entirely. After checking for overfitting using R^2 adjusted, the physical model is also indicated to be superior.
# 
# The Root Mean Squared Error (RMSE) is the most meaningful model metric at face value for our use case. This is because it directly describes how great the error of the model output is in the units of what the model is trying to measure. For this specific data set and physical problem, the RMSE can be interpreted as how likely the coefficient of drag produced by the model is to be away from the real coefficient of drag. The physical model produces a better RMSE than the polynomial, which indicates the superiority of using a physical model again.

# ## Part 1e: Residual Diagnostics
# 
# **Tasks:**
# 1. Compute residuals for best-performing model
# 2. Create diagnostic plots: residuals vs fitted, residuals vs Mach
# 3. Identify outliers visually
# 4. Analyze heteroscedasticity and patterns

# In[ ]:


# Compute residuals for test set
#  Calculate residuals = actual - predicted for your best model
residualsg = y_testg - y_predg
residualsp = y_testp - y_predp


# In[ ]:


# Diagnostic plots
#  Create residuals vs fitted values plot
plt.scatter(y_predg, residualsg, label="Polynomial 10-term")
plt.scatter(y_predp, residualsp, label="Physics 7-term")
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted C_D Values')
plt.ylabel('Residuals')
plt.title('Residual vs. Fitted Values Plot')
plt.grid()
plt.legend()
plt.show()
#  Create residuals vs Mach number plot

# Mach_scaled_min = min(X_gen_scaled[:,2])
# Mach_scaled_max = max(X_gen_scaled[:,2])
# Mach_min = min(X_gen[:,2])
# Mach_max = max(X_gen[:,2])
# print(Mach_scaled_min)
# print(Mach_scaled_max)
# print(Mach_min)
# print(Mach_max)
# scale_factor = (Mach_max-Mach_min)/(Mach_scaled_max-Mach_scaled_min)
# # unscaled_testg,unscaled_testp = [[],[]]
# # st
# # for i in range(len(X_testg[:,2]))
# # np.std
# plt.scatter(((X_testg[:,2]-Mach_scaled_min)*scale_factor) + Mach_min, residualsg, label="Polynomial 10-term")
# plt.scatter(((X_testp[:,2]-Mach_scaled_min)*scale_factor) + Mach_min, residualsp, label="Physics 5-term")
plt.scatter(X_gen_scaled_test[:,1], residualsg, label="Polynomial 10-term")
plt.scatter(X_phys_scaled_test[:,2], residualsp, label="Physics 7-term")
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Scaled Mach Number')
plt.ylabel('Residuals')
plt.title('Residual vs. Scaled Mach Number Plot')
plt.grid()
plt.legend()
plt.show()

plt.scatter(X_gen_scaled_test[:,2], residualsg, label="Re Polynomial 10-term")
plt.scatter(X_phys_scaled_test[:,4], residualsp, label="log(Re) Physics 7-term")
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Scaled log(Reynolds Number)')
plt.ylabel('Residuals')
plt.title('Residual vs. Scaled log(Reynolds Number) Plot')
plt.grid()
plt.legend()
plt.show()
# Optional: Q-Q plot for normality assessment


# ### Analysis: Residual Diagnostics
# 
# ** Write 2 paragraphs discussing:**
# - Pattern in residuals (random or systematic)?
# - Evidence of heteroscedasticity?
# - Which Mach regime has larger residuals?
# - Physical explanation for outliers?
# 
# The pattern in the residuals appears to be entirely random for both the 10-term polynomial and the 7-term physics-based solution. The graphs have an uneven clustering of points due to the nature of random sampling, which produces the visual effect that the graphs have a trend in residuals. Additionally, the graphs don’t appear to exhibit heteroscedasticity to a large degree. Notably, however, it could be interpreted that the Scaled Mach Number Squared Residual Graph has a small degree of heteroscedasticity.
# 
# The points produced by both the polynomial and the physics-based models have clear points past the first-third of the residual graph, where the residuals grow increasingly farther away from the zero line. This seems to indicate that the upper half of the Mach Regime, bordering on and then including the transonic regime, has more advanced physics confounding the model. The current model does a good job of addressing this, but would probably require another term or two to fully address it.

# ## Part 1f: Model Interpretation and Comparison
# 
# **Tasks:**
# 1. Examine coefficients of physics-inspired model
# 2. Interpret coefficients physically
# 3. Compare polynomial vs physics-inspired approaches
# 4. Make final recommendation

# In[ ]:


# Display model coefficients
#  Show coefficients from physics-inspired model (Model B)
print("Final Physics-Based Form:")
print("Scaled Intercept = {}".format(interceptg))
print("Scaled Coefficients = {}".format([i for i in coefficientsg if abs(i) > (10**-16)]))

print("Final Physics-Based Form:")
print("1, a x M, a x Re, M, M^2, log(Re), 1/sqrt(1-M^2)")

print("Scaled Intercept = {}".format(interceptp))
print("Scaled Coefficients = {}".format([i for i in coefficientsp if abs(i) > (10**-16)]))
# print("Coefficients: 1: {}, a: {}, M: {}, Re: {},\na x Re: {}, log(Re): {},\n1/sqrt(1-M^2): {}, (M - M_crit)^2 @ Mcrit > 0.7: {}".format(coefficientsp[0],coefficientsp[1],coefficientsp[2],coefficientsp[3],coefficientsp[4],coefficientsp[5],coefficientsp[6],coefficientsp[7]))
# print("Coefficients: 1:{}, a x Re: {}, M^2: {}, log(Re): {},\n1/sqrt(1-M^2): {}, (M - M_crit)^2 @ Mcrit > 0.7: {}".format(coefficientsp[0],coefficientsp[1],coefficientsp[2],coefficientsp[3],coefficientsp[4],coefficientsp[5]))
# print("Coefficients: 1:{}, a x Re: {}, log(Re): {},\n1/sqrt(1-M^2): {}, (M - M_crit)^2 @ Mcrit > 0.7: {}".format(coefficientsp[0],coefficientsp[1],coefficientsp[2],coefficientsp[3],coefficientsp[4]))

#  Identify largest coefficients by magnitude

# true_coefficientsg = np.zeros(coefficients_numg)
# for i in range(coefficients_numg):
#     true_coefficientsg[i] = coefficientsg[i]/np.std(X_gen[i])

true_coefficientsg = coefficientsg / SD_gen
true_interceptg = interceptg - np.sum((coefficientsg * mean_gen) / SD_gen)
Model_resultsg = (X_gen @ true_coefficientsg) + true_interceptg


# true_coefficientsp = np.zeros(coefficients_nump)
# Model_results = X_phys_scaled
# for i in range(coefficients_num):
#     Model_results[i] = (Model_results[i] * coefficientsp[i] * np.std(Model_results[i])) + np.average(Model_results[i])

# scaler_phys
# beta_o = beta_s / scaler.scale_
# intercept_o = (
#     beta_0
#     - np.sum(beta_s * scaler.mean_ / scaler.scale_)
# )

# y_stdp = np.std(df[c_names[3]])
# for i in range(coefficients_nump):
#     true_coefficientsp[i] = coefficientsp[i]/np.std(X_phys[i])
    # true_coefficientsp[i] = coefficientsp[i]/(max(X_phys_scaled[i])-min(X_phys_scaled[i]))
    # true_coefficientsp[i] = (coefficientsp[i]*(y_stdp/st.stdev(X_phys[i])))
    # true_coefficientsp[i] = (coefficientsp[i]*st.stdev(X_phys_scaled[i]))
    # true_coefficientsp[i] = (coefficientsp[i]*st.stdev(X_phys[i])) + np.average(X_phys[i])
# print("Coefficients: 1:{}, a x Re: {}, M^2: {}, log(Re): {},\n1/sqrt(1-M^2): {}, (M - M_crit)^2 @ Mcrit > 0.7: {}".format(coefficientsp[0],coefficientsp[1],coefficientsp[2],coefficientsp[3],coefficientsp[4],coefficientsp[5]))
# print("Coefficients: 1:{}, a x Re: {}, log(Re): {},\n1/sqrt(1-M^2): {}, (M - M_crit)^2 @ Mcrit > 0.7: {}".format(true_coefficientsp[0],true_coefficientsp[1],true_coefficientsp[2],true_coefficientsp[3],true_coefficientsp[5]))

true_coefficientsp = coefficientsp / SD_phys
true_interceptp = interceptp - np.sum((coefficientsp * mean_phys) / SD_phys)
Model_resultsp = (X_phys @ true_coefficientsp) + true_interceptp

# interceptg,interceptp = [interceptg - sum([true_coefficientsg[k]*np.average(X_gen[k]) for k in range(coefficients_numg)]),
#                  interceptp - sum([true_coefficientsp[k]*np.average(X_phys[k]) for k in range(coefficients_nump)])]

print("\nGeneral Polynomial Intercept = {}".format(true_interceptg))
print("General Polynomial Coefficients: {}".format(true_coefficientsg))
print("General Polynomial Coefficients:\n1: {}, a: {}, M: {}, Re: {},\na^2: {},a x M: {}, a x Re: {},\nM^2: {}, M x Re: {}, Re^2: {}".format(true_interceptg,*true_coefficientsg))
print("Physics-Based Intercept = {}".format(true_interceptp))
print("Physics-Based Coefficients: {}".format(true_coefficientsp))
print("Physics-Based Coefficients:\n1: {}, a x M: {}, a x Re: {}, M: {},\nM^2: {}, log(Re): {}, 1/sqrt(1-M^2): {}".format(true_interceptp,*true_coefficientsp))
# print("The largest coefficient is the intercept, which equals {}".format(interceptp))
# print("The second largest coefficient is then of log(Re), which is equal to {}".format(coefficientsp[1]))
# print("The third largest coefficient is then of 1/sqrt(1-M^2), which is equal to {}".format(coefficientsp[2]))
# print("The fourth largest coefficient is then of (M - M_crit)^2 @ Mcrit > 0.7, which is equal to {}".format(coefficientsp[3]))
# print("The fifth largest coefficient is then of a x Re, which is equal to {}".format(coefficientsp[0]))


# In[ ]:


##### No good explanation online of how to reverse the coefficients into real ones
# # Plots of Real data with the Physics-Based model results
# Model_resultsg = np.zeros((num_rows,1))
# Model_resultsp = np.zeros((num_rows,1))
# for i in list(np.random.randint(0,num_rows-1, size=10)):
#     testval = interceptp + true_coefficientsp[0] + (true_coefficientsp[1]*df[c_names[0]][i]*df[c_names[1]][i]) + (true_coefficientsp[2]*df[c_names[0]][i]*df[c_names[2]][i]) + (true_coefficientsp[3]*(df[c_names[1]][i]**2)) + (true_coefficientsp[4]*np.log(df[c_names[2]][i])) + (true_coefficientsp[5]*(1/((1 - (df[c_names[1]][i]**2))**0.5)))
#     print("Test Index {}: Real C_D = {}, Physics-Based Model C_D = {}, Residual = {}, Percent Error = {}".format(i,df[c_names[3]][i],testval,df[c_names[3]][i]-testval,100*(df[c_names[3]][i]-testval)/df[c_names[3]][i]))
# print("General Polynomial Intercept = {}".format(interceptg))
# print("Physics-Based Intercept = {}".format(interceptp))
# for i in range(num_rows):
#     valg,valp = [interceptg,interceptp]
#     for j in range(coefficients_nump):
#         valp += true_coefficientsp[j]*X_phys[i,j]
#     Model_resultsp[i,0] = valp
#     for j in range(coefficients_numg):
#         valg += true_coefficientsg[j]*X_gen[i,j]
#     Model_resultsg[i,0] = valg

    # if df[c_names[1]][i] > 0.7:
    #     # Model_results[i,0] = interceptp + (true_coefficientsp[0]*df[c_names[0]][i]*df[c_names[2]][i]) + (true_coefficientsp[1]*np.log(df[c_names[2]][i])) + (true_coefficientsp[2]*(1/((1-(df[c_names[1]][i]**2))**0.5))) + (true_coefficientsp[3]*((df[c_names[1]][i] - 0.7)**2))
    #     Model_resultsp[i,0] = interceptp + (true_coefficientsp[1]*df[c_names[0]][i]*df[c_names[2]][i]) + (true_coefficientsp[2]*np.log(df[c_names[2]][i])) + (true_coefficientsp[3]*(df[c_names[1]][i]**2)) + (true_coefficientsp[5]*((df[c_names[1]][i] - 0.7)**2))
    # else:
    #     # Model_results[i,0] = interceptp + (true_coefficientsp[0]*df[c_names[0]][i]*df[c_names[2]][i]) + (true_coefficientsp[1]*np.log(df[c_names[2]][i])) + (true_coefficientsp[2]*(1/((1-(df[c_names[1]][i]**2))**0.5)))
    #     Model_resultsp[i,0] = interceptp + (true_coefficientsp[1]*df[c_names[0]][i]*df[c_names[2]][i]) + (true_coefficientsp[2]*np.log(df[c_names[2]][i])) + (true_coefficientsp[3]*(df[c_names[1]][i]**2))
    # Model_resultsg[i,0] = interceptg + (true_coefficientsg[1]*X_gen[i,1]) + (true_coefficientsg[2]*X_gen[i,2]) + (true_coefficientsg[3]*X_gen[i,3]) + (true_coefficientsg[4]*X_gen[i,4]) + (true_coefficientsg[5]*X_gen[i,5]) + (true_coefficientsg[6]*X_gen[i,6]) + (true_coefficientsg[7]*X_gen[i,7]) + (true_coefficientsg[8]*X_gen[i,8]) + (true_coefficientsg[9]*X_gen[i,9])


# y_stdp = np.std(df[c_names[3]])
# coefficient1 = (10**1)*coefficientsp[1]*y_stdp/np.std(X_phys[1])
# coefficient2 = (10**1)*coefficientsp[2]*y_stdp/np.std(X_phys[2])
# coefficient3 = (10**1)*coefficientsp[3]*y_stdp/np.std(X_phys[3])
# coefficient4 = (10**1)*coefficientsp[4]*y_stdp/np.std(X_phys[4])
# Model_results = np.zeros((num_rows,1))
# for i in range(num_rows):
#     if df[c_names[1]][i] > 0.7:
#         Model_results[i,0] = interceptp + (coefficient1*df[c_names[0]][i]*df[c_names[2]][i]) + (coefficient2*np.log(df[c_names[2]][i])) + (coefficient3*(1/((1-(df[c_names[1]][i]**2))**0.5))) + (coefficient4*((df[c_names[1]][i] - 0.7)**2))
#     else:
#         Model_results[i,0] = interceptp + (coefficient1*df[c_names[0]][i]*df[c_names[2]][i]) + (coefficient2*np.log(df[c_names[2]][i])) + (coefficient3*(1/((1-(df[c_names[1]][i]**2))**0.5)))

# Model_results = np.zeros(num_rows)
# Model_results_columns = np.zeros((num_rows,num_columns))
# for i in range(num_columns):
#     X_phys_scaled[:,i] = X_phys_scaled[:,i]*coefficientsp[i]
# for i in range(num_rows):
#     Model_results[i] = interceptp + sum(X_phys_scaled[i,:])
# Model_results = np.transpose(Model_results)
# print(Model_results[:3])
# print(np.shape(X_phys_scaled))
# print(np.shape(np.array(coefficientsp)))
# print(len(list(df[c_names[0]])))
# print(len(list(Model_results)))
# Model_results = interceptp + (X_phys_scaled*np.array(coefficientsp))
Real_color = "green" # (np.random.random(), np.random.random(), np.random.random())
plt.scatter(df[c_names[0]],df[c_names[-1]],label="Real",color=Real_color)
# xvals,yvals = [list(df[c_names[0]]),list(Model_results)]
# print(len(xvals),len(yvals))
plt.scatter(df[c_names[0]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[0]],Model_resultsg,label="GenP",color="blue")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[1]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[1]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[1]],Model_resultsg,label="GenP",color="blue")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[2]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[2]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[2]],Model_resultsg,label="GenP",color="blue")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()


# ### Interpretation and Recommendation
# 
# ** Write 2-3 paragraphs addressing:**
# 
# **Physical Interpretation:**
# - Do coefficient signs make physical sense?
# - Do magnitudes align with aerodynamic principles?
# 
# **Model Comparison:**
# - Accuracy: test performance winner?
# - Interpretability: which is easier to explain?
# - Generalization: which would you trust for extrapolation?
# - Engineering value: which provides more physical insight?
# 
# **Final Recommendation:**
# - Which model for aircraft performance analysis?
# - Justification based on engineering criteria
# 
# The coefficients all make sense within the scale of the problem. The drag coefficient is less than one and greater than zero, which makes it a relatively small number compared to the angle of attack and Reynolds Number. Therefore, many of the fitting coefficients are extremely small as they involve multiplying multiple higher-order variables together. The standard deviation of each variable also compounds this effect, as once the standardization process is reversed by dividing by the standard deviation, these variables become smaller in magnitude. Examining the signs of the regression coefficients shows that the coefficients all align with their physically understood trends.
# <!-- However, when examining the magnitude, there are clear issues. This is best displayed by the final comparison graphs, which almost entirely fail to capture the negative logarithm effect of the Reynolds Number, the exponential effect of Mach Number, and the relatively constant effect of angle of attack (alpha). This is despite the inclusion of those functions in the model. To force the model to capture these trends, a weighted least-squares solution would have to be implemented. -->
# 
# The physics-based model is clearly a superior predictor of the trends as compared to the general polynomial model.  This is shown through model statistics such as RMSE and R^2. However, it can also be shown visibly as the polynomial model clearly overpredicts the effect of the Reynolds Number at higher Reynolds Numbers. Additionally, the physics-based model can clearly be seen to more closely predict results on the Mach Number graph. The physics-based model is far easier to interpret, as the equations used to create the model correspond to physical effects. Therefore, a user of the model could easily tell which real-world variables most impact the trends in the data. This naturally means that the physics-based model would therefore be much easier for an engineer to trust, as the effects of the model correspond to real physics, making the model far more valuable than just the simple polynomial fit. My final recommendation is to use the physics-based model, as it is a slightly better predictor of the coefficient of drag than the polynomial model, and it uses fewer coefficients. 
# <!-- However, both models are equally mediocre in performance. They both fail to capture visibly apparent trends despite the inclusion and emphasis of those trends in the model. Additionally, both models try to account for these supposedly missing trends, which further reduces the accuracy of the model and produces more outliers.  -->
# <!-- However, I would also recommend that, before implementing the model, some form of weighted least-squares be applied to the physics-based model to fix the issues previously mentioned about how the model fails to correctly apply the equations used to create it. -->

# ---
# # Part 2: Advanced Analysis

# ## Part 2a: Mathematical Derivation
# 
# **Tasks:**
# 1. Derive normal equations using calculus approach
# 2. Derive using geometric approach with projection matrices
# 3. Prove projection matrix properties (symmetric, idempotent)

# ### Derivation: Calculus Approach
# 
# ** Derive the normal equations starting from RSS minimization**
# 
# Starting with:
# $$\text{RSS}(\boldsymbol{\beta}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|^2$$
# 
# Step 1: Expand the quadratic form:
# $$\text{RSS}(\boldsymbol{\beta}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|^2 = (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta})^{T}(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}) = \boldsymbol{y}^{T}\boldsymbol{y} - 2\boldsymbol{\beta}^{T}\boldsymbol{X}^{T}\boldsymbol{y} + \boldsymbol{\beta}^{T}\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}$$
# 
# Step 2: Differentiate RSS with respect to Beta & set equal to zero:
# $$\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\boldsymbol{X}^{T}\boldsymbol{y} + 2\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta} = 0$$
# 
# Step 3: Simplify the derivative of RSS with respect to Beta when set to zero:
# $$\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}^* = \boldsymbol{X}^{T}\boldsymbol{y}$$
# 
# Step 4: Final result (if X^T*X is invertible):
# $$\boldsymbol{\beta}^* = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

# ### Derivation: Geometric Approach
# 
# ** Derive using projection matrix and orthogonality**
# 
# Define projection matrix:
# $$\boldsymbol{P} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$$
# 
# Prove properties:
# 1. Symmetric: $\boldsymbol{P}^T = \boldsymbol{P}$
# 2. Idempotent: $\boldsymbol{P}^2 = \boldsymbol{P}$
# 
# Symmetric Proof:
# 
# Step 1: Apply transpose to both sides and simplify:
# $$\boldsymbol{P}^T = (\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)^T = (\boldsymbol{X})^T((\boldsymbol{X}^T\boldsymbol{X})^{-1})^T(\boldsymbol{X}^T)^T = \boldsymbol{X}^T((\boldsymbol{X}^T\boldsymbol{X})^{-1})^T\boldsymbol{X}$$
# 
# Step 2: Move the transpose inside of the inverse:
# $$\boldsymbol{P}^T = \boldsymbol{X}^T((\boldsymbol{X}^T\boldsymbol{X})^{-1})^T\boldsymbol{X} = \boldsymbol{X}^T(((\boldsymbol{X}^T)^T(\boldsymbol{X})^T)^{-1})\boldsymbol{X} = \boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{X}^T)^{-1}\boldsymbol{X}$$
# 
# Step 3: Set P^T equal to P:
# $$\boldsymbol{P}^T = \boldsymbol{P} = \boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{X}^T)^{-1}\boldsymbol{X} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T $$
# 
# Step 4: Move X^T from left to right by inversing:
# $$(\boldsymbol{X}\boldsymbol{X}^T)^{-1}\boldsymbol{X} = (\boldsymbol{X}^T)^{-1}\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T $$
# 
# Step 5: Move (X*X^T)^-1 from left to right by inversing:
# $$\boldsymbol{X} = \boldsymbol{X}\boldsymbol{X}^T(\boldsymbol{X}^T)^{-1}\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T $$
# 
# Step 6: Reduce and then simplify the right side of the equation:
# $$\boldsymbol{X} = \boldsymbol{X}\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$$
# 
# Step 7: Move X^T from right to left by inversing:
# $$\boldsymbol{X}(\boldsymbol{X}^T)^{-1} = \boldsymbol{X}\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T(\boldsymbol{X}^T)^{-1}$$
# 
# Step 8: Reduce the right side then move (X^T*X)^-1 from right to left by inversing:
# $$\boldsymbol{X}(\boldsymbol{X}^T)^{-1}(\boldsymbol{X}^T\boldsymbol{X}) = \boldsymbol{X}\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}(\boldsymbol{X}^T\boldsymbol{X})$$
# 
# Step 9: Reduce the left and right sides:
# $$\boldsymbol{X}\boldsymbol{X} = \boldsymbol{X}\boldsymbol{X}$$
# 
# Step 10: Final Step:
# $$\boldsymbol{X}^{-1}\boldsymbol{X}^{-1}\boldsymbol{X}\boldsymbol{X} = \boldsymbol{X}^{-1}\boldsymbol{X}^{-1}\boldsymbol{X}\boldsymbol{X}$$
# $$1 = 1$$
# $$\boldsymbol{P}^T = \boldsymbol{P}$$
# 
# Idempotent Proof:
# 
# Step 1: Multiply P by itself:
# <!-- $$\boldsymbol{P}^2 = \boldsymbol{P}\boldsymbol{P} = \boldsymbol{P}\boldsymbol{P}^T = (\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)(\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{X}^T)^{-1}\boldsymbol{X})$$ -->
# $$\boldsymbol{P}^2 = \boldsymbol{P}\boldsymbol{P} = (\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)$$
# 
# Step 2: Regroup terms:
# $$\boldsymbol{P}^2 = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}(\boldsymbol{X}^T\boldsymbol{X})(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$$
# 
# Step 3: Reduce terms on the right side:
# $$\boldsymbol{P}^2 = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T = \boldsymbol{P}$$
# 
# Step 4: Final step:
# $$\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$$
# $$1 = 1$$
# $$\boldsymbol{P}^2 = \boldsymbol{P}$$
# <!-- $$\boldsymbol{P}^2 = \boldsymbol{P} = (\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)(\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{X}^T)^{-1}\boldsymbol{X}) = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$$ -->
# <!-- (\boldsymbol{X}(\boldsymbol{X})^{-1}(\boldsymbol{X}^T)^{-1}\boldsymbol{X}^T)(\boldsymbol{X}^T(\boldsymbol{X}^T)^{-1}(\boldsymbol{X})^{-1}\boldsymbol{X}) -->
# <!-- $$(\boldsymbol{X}\boldsymbol{X}^T)(\boldsymbol{X}^T)^{-1}(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T) = \boldsymbol{X}$$
# 
# $$\boldsymbol{X}(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T) = \boldsymbol{X}$$
# 
# $$(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T) = \boldsymbol{I}$$
# 
# $$(\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T) = (\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T)^{-1}$$ -->
# 
# Geometric interpretation and derivation:
# 
# Normal equations:
# $$\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}^* = \boldsymbol{X}^{T}\boldsymbol{y}$$
# 
# If X^T*X is invertible:
# $$\boldsymbol{\beta}^* = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$
# 
# Error (residual) definition:
# $$\boldsymbol{r} = \boldsymbol{y} - \hat{\boldsymbol{y}} = \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}$$
# 
# Goal of minimizing the length of the error vector:
# $$\underset{\boldsymbol{\beta}}{\text{min}}\parallel\boldsymbol{r}\parallel^2 = \underset{\boldsymbol{\beta}}{\text{min}}\parallel\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\parallel^2$$
# 
# Geometrically the error must be perpendicular to the columns of X:
# $$\boldsymbol{X}^{T}\boldsymbol{r} = \boldsymbol{X}^{T}(\boldsymbol{y} - \hat{\boldsymbol{y}}) = 0$$
# 
# Equations for optimal Beta:
# $$\hat{\boldsymbol{y}}^* = \boldsymbol{X}\boldsymbol{\beta}^*$$
# $$\boldsymbol{r}^* = \boldsymbol{y} - \hat{\boldsymbol{y}}^*$$
# 
# Substituting Equations:
# $$\boldsymbol{X}^{T}(\boldsymbol{y} - \hat{\boldsymbol{y}}^*) = \boldsymbol{X}^{T}(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*) = 0$$
# 
# Rearranging the terms produces the normal equations:
# $$\boldsymbol{X}^{T}\boldsymbol{y} - \boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}^* = 0$$
# $$\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}^* = \boldsymbol{X}^{T}\boldsymbol{y}$$
# 
# If X^T*X is invertible:
# $$\boldsymbol{\beta}^* = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

# ## Part 2b: Gradient Descent Comparison
# 
# **Tasks:**
# 1. Implement gradient descent using SGDRegressor
# 2. Compare to closed-form solution
# 3. Plot convergence
# 4. Discuss trade-offs

# In[ ]:


# Gradient Descent Implementation
#  Use SGDRegressor with your best feature set
# # Redirect stdout
# old_stdout = sys.stdout
# sys.stdout = mystdout = StringIO()

iterations = 1000
# eta0_val = 0.001
# learning_rates = ["constant","optimal","invscaling","adaptive","pa1","pa2"]
learning_rates = [("constant",0.005),("optimal",0.005),("invscaling",0.005),("adaptive",0.005),("constant",0.001),("optimal",0.001),("invscaling",0.001),("adaptive",0.001)]
colors = ["black","blue","red","green","orange","purple","pink","yellow"]
SGD_data = {}
for i in learning_rates:
    SGD_data[i] = {"iteration": [], "cost": [], "coefficients": [], "intercept": []}
    if (i[0] == "pa1") or ([0] == "pa2"):
        SGD_phys = SGDRegressor(verbose=0, eta0 = i[1], random_state=42, learning_rate = i[0], loss="epsilon_insensitive") # , max_iter=1000, tol=1e-4
    else:
        SGD_phys = SGDRegressor(verbose=0, eta0 = i[1], random_state=42, learning_rate = i[0])
    for j in range(iterations):
        # X_trainp, X_testp, y_trainp, y_testp = train_test_split(X_phys, y, test_size=0.2, random_state=j)
        # scaler_phys = StandardScaler()
        # X_phys_scaled_train = scaler_phys.fit_transform(X_trainp)
        # X_phys_scaled_test = scaler_phys.transform(X_testp)

        X_shuff, y_shuff = shuffle(X_phys_scaled_train, y_trainp, random_state=j)
        SGD_phys.partial_fit(X_shuff, y_shuff)
        # SGD_phys.partial_fit(X_trainp, y_trainp)
        SGD_data[i]["iteration"].append(j)
        # SGD_data[i]["cost"].append(np.mean((y_testp - SGD_phys.predict(X_phys_scaled_test))**2)) # mean of R^2
        ypred_iter = SGD_phys.predict(X_phys_scaled_test)
        SGD_data[i]["cost"].append(mean_squared_error(y_testp, ypred_iter)) # "MSE"
        # SGD_data[i]["cost"].append(mean_squared_error(y_testp, SGD_phys.predict(X_phys_scaled_test))*0.5) # "MSE/2"
        SGD_data[i]["coefficients"].append(SGD_phys.coef_.copy())
        SGD_data[i]["intercept"].append(SGD_phys.intercept_.item())
    # y_SGD_predp_final = SGD_phys.predict(X_phys_scaled_test) # 1 greater than the iteration number
    SGD_data[i]["Final MSE"] = mean_squared_error(y_testp, ypred_iter)
    SGD_data[i]["Final R^2"] = r2_score(y_testp, ypred_iter)
    SGD_data[i]["Final MAE"] = mean_absolute_error(y_testp, ypred_iter)
    SGD_data[i]["Final MAPE"] = mean_absolute_percentage_error(y_testp, ypred_iter)
    SGD_data[i]["Final Max Error"] = max_error(y_testp, ypred_iter)
    # SGD_data[i]["Final CV Score"] = max_error(SGDRegressor(),X_phys_scaled,y)
    # print("Final R^2 of the SGD learning rate of {} = {}".format(i,SGD_data[i]["cost"][-1]))


###### Different methods for looking at improvement per iteration
# SGD_phys.fit(X_trainp,y_trainp)

# # Restore stdout
# sys.stdout = old_stdout

# y_SGDpredp = modelp.predict(X_testp)

# # loss_curve_ is a list of average losses per epoch
# losses = SGD_phys.loss_curve_

# # Print the losses
# for epoch, loss in enumerate(losses):
#     print(f"Epoch {epoch + 1}: Loss = {loss}")

# interceptp = modelp.intercept_
# coefficientsp = modelp.coef_
# coefficients_nump = len(coefficientsp)
######

#  Experiment with learning rates and iterations
#  Track loss over iterations
for i in range(len(learning_rates)):
    plt.plot(SGD_data[learning_rates[i]]["iteration"],SGD_data[learning_rates[i]]["cost"],label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("Cost: Mean Squared Error (MSE)")
plt.title("MSE vs. SGD Iterations")
plt.legend()
plt.show()

for i in range(len(learning_rates)):
    plt.plot(SGD_data[learning_rates[i]]["iteration"],np.log10(SGD_data[learning_rates[i]]["cost"]),label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("log10(Cost: Mean Squared Error (MSE))")
plt.title("log10(MSE) vs. SGD Iterations")
plt.legend()
plt.show()

for i in range(len(learning_rates)):
    yvals = [SGD_data[learning_rates[i]]["cost"][j+1] - SGD_data[learning_rates[i]]["cost"][j] for j in range(iterations - 1)]
    plt.plot(SGD_data[learning_rates[i]]["iteration"][:-1],yvals,label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("Cost: Mean Squared Error (MSE) Rate of Change")
plt.title("MSE Rate of Change vs. SGD Iterations")
plt.legend()
plt.show()

for i in range(len(learning_rates)):
    yvals = np.log10([abs(SGD_data[learning_rates[i]]["cost"][j+1] - SGD_data[learning_rates[i]]["cost"][j]) for j in range(iterations - 1)])
    # print(yvals)
    # for j in range(len(yvals)):
    #     if yvals[j] != 0:
    #         yvals[j] = np.log10(yvals[j])
    plt.plot(SGD_data[learning_rates[i]]["iteration"][:-1],yvals,label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("log10(abs(Cost: Mean Squared Error (MSE) Rate of Change)")
plt.title("log10(abs(MSE Rate of Change)) vs. SGD Iterations")
plt.legend()
plt.show()

for i in [0,2,3,4,6,7]:
    plt.plot(SGD_data[learning_rates[i]]["iteration"],SGD_data[learning_rates[i]]["cost"],label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("Cost: Mean Squared Error (MSE)")
plt.title("MSE vs. SGD Iterations w/o optimal")
plt.legend()
plt.show()

for i in [0,2,3,4,6,7]:
    plt.plot(SGD_data[learning_rates[i]]["iteration"],np.log10(SGD_data[learning_rates[i]]["cost"]),label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("log10(Cost: Mean Squared Error (MSE))")
plt.title("log10(MSE) vs. SGD Iterations w/o optimal")
plt.legend()
plt.show()

for i in [0,2,3,4,6,7]:
    yvals = [SGD_data[learning_rates[i]]["cost"][j+1] - SGD_data[learning_rates[i]]["cost"][j] for j in range(iterations - 1)]
    plt.plot(SGD_data[learning_rates[i]]["iteration"][:-1],yvals,label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("Cost: Mean Squared Error (MSE) Rate of Change")
plt.title("MSE Rate of Change vs. SGD Iterations w/o optimal")
plt.legend()
plt.show()

for i in [0,2,3,4,6,7]:
    yvals = np.log10([abs(SGD_data[learning_rates[i]]["cost"][j+1] - SGD_data[learning_rates[i]]["cost"][j]) for j in range(iterations - 1)])
    # print(yvals)
    # for j in range(len(yvals)):
    #     if yvals[j] != 0:
    #         yvals[j] = np.log10(yvals[j])
    plt.plot(SGD_data[learning_rates[i]]["iteration"][:-1],yvals,label = learning_rates[i],color = colors[i])
plt.grid()
plt.xlabel("SGD Iterations")
plt.ylabel("log10(abs(Cost: Mean Squared Error (MSE) Rate of Change)")
plt.title("log10(abs(MSE Rate of Change)) vs. SGD Iterations w/o optimal")
plt.legend()
plt.show()


# In[ ]:


# Compare coefficients: GD vs closed-form
#  Display both sets of coefficients side by side
msepSGD = SGD_data[("invscaling",0.001)]["Final MSE"]
rmsepSGD = np.sqrt(msepSGD)
r2pSGD = SGD_data[("invscaling",0.001)]["Final R^2"]
r2apSGD = 1 - ((1-r2pSGD)*(n_phys-1)/(n_phys - (coefficients_nump + 1) - 1)) # plus 1 for d to account for intercept
maepSGD = SGD_data[("invscaling",0.001)]["Final MAE"]
mapepSGD = SGD_data[("invscaling",0.001)]["Final MAPE"]
maxepSGD = SGD_data[("invscaling",0.001)]["Final Max Error"]
# cvsp = cross_val_score(LinearRegression(),X_phys_scaled,y)

reg_metrics["PB_SGD"] = {"MSE":msepSGD,"RMSE":rmsepSGD,"R^2":r2pSGD,"R_adj^2":r2apSGD,"MAE":maepSGD,"MAPE":mapepSGD,"Max Error":maxepSGD}

print("Key Metrics: MSE,                  RMSE,                  R^2,                R_adj^2,            MAE,                   MAPE,                 Max Error")
print("GP:          {}, {}, {}, {}, {}, {}, {}".format(mseg,rmseg,r2g,r2ag,maeg,mapeg,maxeg))
print("PB:          {}, {}, {}, {}, {}, {}, {}".format(msep,rmsep,r2p,r2ap,maep,mapep,maxep))
print("PB_SGD:      {}, {}, {}, {}, {}, {}, {}".format(msepSGD,rmsepSGD,r2pSGD,r2apSGD,maepSGD,mapepSGD,maxepSGD))
print()

interceptp_SGD = SGD_data[("invscaling",0.001)]["intercept"][-1]
coefficientsp_SGD = SGD_data[("invscaling",0.001)]["coefficients"][-1]
# true_coefficientsp_SGD = np.zeros(coefficients_nump)
# for i in range(coefficients_nump):
#     true_coefficientsp_SGD[i] = coefficientsp_SGD[i]/np.std(X_phys[i])
# true_interceptp_SGD = interceptp_SGD - sum([true_coefficientsp_SGD[k]*np.average(X_phys[k]) for k in range(coefficients_nump)])
# Model_resultsp_SGD = np.zeros((num_rows,1))
# for i in range(num_rows):
#     valp_SGD = true_interceptp_SGD
#     for j in range(coefficients_nump):
#         valp_SGD += true_coefficientsp_SGD[j]*X_phys[i,j]
#     Model_resultsp_SGD[i,0] = valp_SGD

true_coefficientsp_SGD = coefficientsp_SGD / SD_phys
true_interceptp_SGD = interceptp_SGD - np.sum((coefficientsp_SGD * mean_phys) / SD_phys)
Model_resultsp_SGD = (X_phys @ true_coefficientsp_SGD) + true_interceptp_SGD

print("Physics-Based Intercept = {}".format(true_interceptp))
print("Physics-Based Coefficients: {}".format(true_coefficientsp))
print("Physics-Based Coefficients:\n1: {}, a x M: {}, a x Re: {}, M: {},\nM^2: {}, log(Re): {}, 1/sqrt(1-M^2): {}".format(true_interceptp,*true_coefficientsp))

print("Physics-Based SGD Intercept = {}".format(true_interceptp_SGD))
print("Physics-Based SGD Coefficients: {}".format(true_coefficientsp_SGD))
print("Physics-Based SGD Coefficients:\n1: {}, a x M: {}, a x Re: {}, M: {},\nM^2: {}, log(Re): {}, 1/sqrt(1-M^2): {}".format(true_interceptp_SGD,*true_coefficientsp_SGD))
# for i in list(np.random.randint(0,num_rows-1, size=10)):
#     testval = true_interceptp_SGD + true_coefficientsp_SGD[0] + (true_coefficientsp_SGD[1]*df[c_names[0]][i]*df[c_names[1]][i]) + (true_coefficientsp_SGD[2]*df[c_names[0]][i]*df[c_names[2]][i]) + (true_coefficientsp_SGD[3]*(df[c_names[1]][i]**2)) + (true_coefficientsp_SGD[4]*np.log(df[c_names[2]][i])) + (true_coefficientsp_SGD[5]*(1/((1 - (df[c_names[1]][i]**2))**0.5)))
#     print("Test Index {}: Real C_D = {}, Physics-Based Model C_D = {}, Residual = {}, Percent Error = {}".format(i,df[c_names[3]][i],testval,df[c_names[3]][i]-testval,100*(df[c_names[3]][i]-testval)/df[c_names[3]][i]))



# In[ ]:


# Convergence plot
#  Plot loss vs iteration number

Real_color = "green" # (np.random.random(), np.random.random(), np.random.random())
plt.scatter(df[c_names[0]],df[c_names[-1]],label="Real",color=Real_color)
# xvals,yvals = [list(df[c_names[0]]),list(Model_results)]
# print(len(xvals),len(yvals))
plt.scatter(df[c_names[0]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[0]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[0]],Model_resultsp_SGD,label="PhsM SGD",color="orange")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[1]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[1]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[1]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[1]],Model_resultsp_SGD,label="PhsM SGD",color="orange")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[2]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[2]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[2]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[2]],Model_resultsp_SGD,label="PhsM SGD",color="orange")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()


# ### Analysis: Gradient Descent vs Closed-Form
# 
# ** Write 2 paragraphs discussing:**
# - Convergence behavior and iterations needed
# - When is GD preferred over closed-form?
# - Computational complexity differences (O(d³) for matrix inversion)
# - Most appropriate approach for this dataset
# 
# The SGD converged quickly towards a low Mean Squared Error (MSE) with all the algorithms except “optimal”. With optimal for some reason the error was multiple orders of magnitude higher than the rest. The best and most consistently performing algorithm was “invscaling” which took longer to converge than “adaptive” in most cases but was far more stable than “adaptive” in most test cases. The reason this method, Stochastic Gradient Descent (SGD), is often used over the closed-form solution is because many regression algorithms have hundreds of fitting functions that are incorporated together. This causes the closed-form solution to have a computational complexity requirement on the order of O(d^3), while SGD has a computational requirement complexity of roughly O(nd). This can drastically reduce computation time in some applications that have a large number of fitting functions.
# 
# For this data set there are over two hundred rows of data. Knowing that we are fitting seven parameters (six if you ignore the intercept), the cost of inverting the matrix is low relative to the computation time of SGD. To put it simply, this is because the matrix has many more rows than columns. Some additional notes are that if the time step for the SGD algorithm (eta0) is too high the solution will struggle to reach the exact optimal solution and will instead oscillate around it. To keep “eta0” low the “adaptive” algorithm is a good alternative although it can be potentially more inconsistent at times.
# 

# ## Part 2c: Multicollinearity Analysis
# 
# **Tasks:**
# 1. Create correlation heatmap for all features
# 2. Calculate VIF for each feature
# 3. Identify and discuss multicollinearity issues

# In[ ]:


# Correlation heatmap for physics-inspired features
#  Create heatmap showing pairwise correlations
df_X_gen = pd.DataFrame(X_gen_scaled_train)
co_mtx = df_X_gen.corr(numeric_only=True)
print(co_mtx)
sns.heatmap(co_mtx,cmap="YlGnBu", annot=True)
plt.show()

df_X_Phys = pd.DataFrame(X_phys_scaled_train)
co_mtx = df_X_Phys.corr(numeric_only=True)
print(co_mtx)
sns.heatmap(co_mtx,cmap="YlGnBu", annot=True)
plt.show()


# In[ ]:


# Calculate VIF
#  Compute variance inflation factor for each feature
# Hint: Use statsmodels.stats.outliers_influence.variance_inflation_factor
print("VIF of Original Data Frame:")
X = add_constant(df)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

print("VIF of General Polynomial Model Features Data Frame:")
X = add_constant(df_X_gen)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

print("VIF of Physics-Based Model Features Data Frame:")
X = add_constant(df_X_Phys)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# ### Analysis: Multicollinearity
# 
# ** Write 2 paragraphs discussing:**
# - Which features have high correlation (|ρ| > 0.7) or VIF > 10?
# - Physical explanation for multicollinearity (are terms naturally related?)
# - Impact on coefficient interpretability, prediction accuracy, stability
# - Should any features be removed? Why or why not?
# 
# The general polynomial has far more terms than the physics-based model if the intercept of both aren’t considered. Therefore, the heatmap for the general polynomial data is slightly difficult to interpret, however, fitting terms that relate data from multiple columns clearly have high degrees of correlation on the heatmap. There are some more interesting trends in the physics-based model because it’s easier to read and also because there are some additional non-linear terms included. All the terms consisting of just Mach Number are highly correlated. However, the interesting relations come from the correlation between Mach Number and alpha. All the Mach Number terms have a correlation of roughly 0.25 with the alpha * Mach Number term. This may indicate a not so insignificant relationship.
# 
# Looking at a different metric, the Variance Inflation Factor (VIF), it can be seen that many of the terms in the general polynomial have a value greater than ten. This indicates that many of the terms have a problematic level of multicollinearity. High levels of multicollinearity lead to models that are more difficult to understand because of the inflation of standard error and generally a risk of overfitting. Looking at the physics-based model, there are even higher level of multicollinearity, which is generally not a good sign. By the time I realized the issue of multi-collinearity it was too late for me to go and fix it. Therefore, my recommendation, if I had a chance to change the model, would be to remove the linear Mach Number term. There isn’t a strong argument from a physics standpoint of including it beyond a few trace equations. Mach Number squared is an infinitely better predictor of the data that can actually be correlated to a key physical effect. There is also the debate that the compressibility term be removed because of its high VIF. However, I feel like some other variables should be added back to the model, such as alpha^2 and other terms that correspond to physical phenomena before compressibility is removed. The result of adding more terms to the model, like with the general polynomial, is that VIF generally decreases if done correctly. Therefore, the VIF may decrease enough for the term to stay in the regression.
# 

# ## Part 2d: Per-Regime Analysis and Stability
# 
# **Tasks:**
# 1. Train separate models for subsonic and transonic regimes
# 2. Compare regime-specific vs global model performance
# 3. Identify dominant features in each regime
# 4. Assess coefficient stability via cross-validation

# In[ ]:


# Regime-specific modeling
#  Split data into subsonic (M < 0.7) and transonic (M >= 0.7)

# dfsorted = df.copy(deep=True)
# print(dfsorted)
# sorted_df = df.sort_values(by="Mach")
# sorted_indices = sorted_df.index
# sorted_indices = sorted_indices.tolist()
# # print(sorted_indices)
# ind = 0
# while df[c_names[1]][sorted_indices[ind]] < 0.7:
#     # print(dfsorted[c_names[1]][ind])
#     ind += 1
# print(ind)
# # print(df[c_names[1]][sorted_indices[ind]])
# # print(df[c_names[1]][sorted_indices[num_rows - 12:]])
# sorted_indices_sub = sorted_indices[:ind]
# sorted_indices_tra = sorted_indices[ind:]
# df_sub = df.iloc[sorted_indices_sub]
# df_tra = df.iloc[sorted_indices_tra]
df_sub = df[df[c_names[1]] < 0.7].copy()
df_tra = df[df[c_names[1]] >= 0.7].copy()
y_sub = np.array(df_sub[c_names[3]])
y_tra = np.array(df_tra[c_names[3]])
# print(dfsub)
# print(dftra)
num_rows_sub = len(df_sub[c_names[1]][:])
num_rows_tra = len(df_tra[c_names[1]][:])
# print()
# print(num_rows_sub)
# print(num_rows_tra)

####
#  Train separate models for each regime
X_phys_sub = np.zeros((num_rows_sub, 6))
# X_phys_sub[:,0] = [1 for i in range(num_rows_sub)] # 1
X_phys_sub[:,0] = df_sub[c_names[0]]*df_sub[c_names[1]] # a x M
X_phys_sub[:,1] = df_sub[c_names[0]]*df_sub[c_names[2]] # a x Re
X_phys_sub[:,2] = df_sub[c_names[1]] # M
# X_phys_sub[:,2] = np.log(df[c_names[1]]) # log(M^2)
X_phys_sub[:,3] = df_sub[c_names[1]]**2 # M^2
X_phys_sub[:,4] = np.log(df_sub[c_names[2]]) # log(Re)
# X_phys_sub[:,4] = df[c_names[2]]**(-0.2) # log(Re)
X_phys_sub[:,5] = 1/((1 - (df_sub[c_names[1]]**2))**0.5) # 1/sqrt(1-M^2)

# print(df[c_names[1]][0]*df[c_names[2]][0])
print("First Row of Physics-Based Fit Matrix: ", X_phys_sub[0,:])
# print("Last Row of Physics-Based Fit Matrix: ", X_phys_sub[-1,:])

X_trainp_sub, X_testp_sub, y_trainp_sub, y_testp_sub = train_test_split(X_phys_sub, y_sub, test_size=0.2, shuffle=True, random_state=42)

scaler_phys_sub = StandardScaler()
X_phys_sub_scaled_train = scaler_phys_sub.fit_transform(X_trainp_sub)
X_phys_sub_scaled_test = scaler_phys_sub.transform(X_testp_sub)
mean_sub = scaler_phys_sub.mean_
SD_sub = scaler_phys_sub.scale_

#  Train separate models for each regime
X_phys_tra = np.zeros((num_rows_tra, 7))
# X_phys_tra[:,0] = [1 for i in range(num_rows_tra)] # 1
X_phys_tra[:,0] = df_tra[c_names[0]]*df_tra[c_names[1]] # a x M
X_phys_tra[:,1] = df_tra[c_names[0]]*df_tra[c_names[2]] # a x Re
X_phys_tra[:,2] = df_tra[c_names[1]] # M
# X_phys_tra[:,2] = np.log(df[c_names[1]]) # log(M^2)
X_phys_tra[:,3] = df_tra[c_names[1]]**2 # M^2
X_phys_tra[:,4] = np.log(df_tra[c_names[2]]) # log(Re)
# X_phys_tra[:,4] = df[c_names[2]]**(-0.2) # log(Re)
X_phys_tra[:,5] = 1/((1 - (df_tra[c_names[1]]**2))**0.5) # 1/sqrt(1-M^2)
# print(df_tra[c_names[1]])
for i in range(num_rows_tra):
    Mach = df_tra[c_names[1]].iloc[i]
    if Mach > 0.7:
        X_phys_tra[i,6] = (Mach - 0.7)**2 # (M - M_crit)^2
    else:
        X_phys_tra[i,6] = 0

# print(df[c_names[1]][0]*df[c_names[2]][0])
print("First Row of Physics-Based Fit Matrix: ", X_phys_tra[0,:])
# print("Last Row of Physics-Based Fit Matrix: ", X_phys_tra[-1,:])
#  Apply StandardScaler

X_trainp_tra, X_testp_tra, y_trainp_tra, y_testp_tra = train_test_split(X_phys_tra, y_tra, test_size=0.2, shuffle=True, random_state=42)

scaler_phys_tra = StandardScaler()
X_phys_tra_scaled_train = scaler_phys_tra.fit_transform(X_trainp_tra)
X_phys_tra_scaled_test = scaler_phys_tra.transform(X_testp_tra)
mean_tra = scaler_phys_tra.mean_
SD_tra = scaler_phys_tra.scale_

####
modelp_sub = LinearRegression()
modelp_sub.fit(X_phys_sub_scaled_train, y_trainp_sub)
modelp_tra = LinearRegression()
modelp_tra.fit(X_phys_tra_scaled_train, y_trainp_tra)

y_predp_sub = modelp_sub.predict(X_phys_sub_scaled_test)
y_predp_tra = modelp_tra.predict(X_phys_tra_scaled_test)


# In[ ]:


#  Compare performance to global model
interceptp_sub = modelp_sub.intercept_
coefficientsp_sub = modelp_sub.coef_
coefficients_nump_sub = len(coefficientsp_sub)

n_sub = len(y_testp_sub)
msep_sub = mean_squared_error(y_testp_sub, y_predp_sub)
rmsep_sub = np.sqrt(msep_sub)
r2p_sub = r2_score(y_testp_sub, y_predp_sub)
r2ap_sub = 1 - ((1-r2p_sub)*(n_sub-1)/(n_sub - (coefficients_nump_sub + 1) - 1)) # plus 1 for d to account for intercept
maep_sub = mean_absolute_error(y_testp_sub, y_predp_sub)
mapep_sub = mean_absolute_percentage_error(y_testp_sub, y_predp_sub)
maxep_sub = max_error(y_testp_sub, y_predp_sub)
# cvsp_sub = cross_val_score(LinearRegression(),X_phys_sub_scaled_train,y_testp_sub)

interceptp_tra = modelp_tra.intercept_
coefficientsp_tra = modelp_tra.coef_
coefficients_nump_tra = len(coefficientsp_tra)

n_tra = len(y_testp_tra)
msep_tra = mean_squared_error(y_testp_tra, y_predp_tra)
rmsep_tra = np.sqrt(msep_tra)
r2p_tra = r2_score(y_testp_tra, y_predp_tra)
r2ap_tra = 1 - ((1-r2p_tra)*(n_tra-1)/(n_tra - (coefficients_nump_tra-1) - 1)) # plus 1 for d to account for intercept
maep_tra = mean_absolute_error(y_testp_tra, y_predp_tra)
mapep_tra = mean_absolute_percentage_error(y_testp_tra, y_predp_tra)
maxep_tra = max_error(y_testp_tra, y_predp_tra)
# cvsp_tra = cross_val_score(LinearRegression(),X_phys_tra_scaled_train,y_testp_tra)

print("Key Metrics:          MSE,                  RMSE,                  R^2,                R_adj^2,            MAE,                   MAPE,                 Max Error")
print("PB:                   {}, {}, {}, {}, {}, {}, {}".format(msep,rmsep,r2p,r2ap,maep,mapep,maxep))
print("PB Subsonic:          {}, {}, {}, {}, {}, {}, {}".format(msep_sub,rmsep_sub,r2p_sub,r2ap_sub,maep_sub,mapep_sub,maxep_sub))
print("PB Transonic:         {}, {}, {}, {}, {}, {}, {}".format(msep_tra,rmsep_tra,r2p_tra,r2ap_tra,maep_tra,mapep_tra,maxep_tra))

coefficients_nump_sub = len(X_phys_sub_scaled_train[0,:])
# print(coefficients_nump_sub)
true_coefficientsp_sub = coefficientsp_sub / SD_sub
true_interceptp_sub = interceptp_sub - np.sum((coefficientsp_sub * mean_sub) / SD_sub)
# true_coefficientsp_sub = np.zeros(coefficients_nump_sub)
# for i in range(coefficients_nump_sub):
#     true_coefficientsp_sub[i] = coefficientsp_sub[i]/np.std(X_phys_sub[:,i])
# true_interceptp_sub = interceptp_sub - sum([true_coefficientsp_sub[k]*np.mean(X_phys_sub[:,k]) for k in range(coefficients_nump_sub)])
# Model_resultsp_sub = np.zeros((num_rows_sub,1))
# for i in range(num_rows_sub):
#     valp_sub = true_interceptp_sub
#     for j in range(coefficients_nump_sub):
#         valp_sub += true_coefficientsp_sub[j]*X_phys_sub[i,j]
#     Model_resultsp_sub[i,0] = valp_sub
## faster
Model_resultsp_sub = (X_phys_sub @ true_coefficientsp_sub) + true_interceptp_sub

true_coefficientsp_tra = coefficientsp_tra / SD_tra
true_interceptp_tra = interceptp_tra - np.sum((coefficientsp_tra * mean_tra) / SD_tra)
coefficients_nump_tra = len(X_phys_tra_scaled_train[0,:])
# true_coefficientsp_tra = np.zeros(coefficients_nump_tra)
# for i in range(coefficients_nump_tra):
#     true_coefficientsp_tra[i] = coefficientsp_tra[i]/np.std(X_phys_tra[:,i])
# true_interceptp_tra = interceptp_tra - sum([true_coefficientsp_tra[k]*np.mean(X_phys_tra[:,k]) for k in range(coefficients_nump_tra)])
# Model_resultsp_tra = np.zeros((num_rows_tra,1))
# for i in range(num_rows_tra):
#     valp_tra = true_interceptp_tra
#     for j in range(coefficients_nump_tra):
#         valp_tra += true_coefficientsp_tra[j]*X_phys_tra[i,j]
#     Model_resultsp_tra[i,0] = valp_tra
## faster
Model_resultsp_tra = (X_phys_tra @ true_coefficientsp_tra) + true_interceptp_tra

print("Scaled Physics-Based Subsonic Intercept = {}".format(interceptp_sub))
print("Scaled Physics-Based Subsonic Coefficients: {}".format(coefficientsp_sub))
print("Scaled Physics-Based Transonic Intercept = {}".format(interceptp_tra))
print("Scaled Physics-Based Transonic Coefficients: {}".format(coefficientsp_tra))
print()
print("Physics-Based Subsonic Intercept = {}".format(true_interceptp_sub))
print("Physics-Based Subsonic Coefficients: {}".format(true_coefficientsp_sub))
print("Physics-Based Transonic Intercept = {}".format(true_interceptp_tra))
print("Physics-Based Transonic Coefficients: {}".format(true_coefficientsp_tra))


# In[ ]:


Real_color = "green" # (np.random.random(), np.random.random(), np.random.random())
plt.scatter(df[c_names[0]],df[c_names[-1]],label="Real",color=Real_color)
# xvals,yvals = [list(df[c_names[0]]),list(Model_results)]
# print(len(xvals),len(yvals))
plt.scatter(df[c_names[0]],Model_resultsp,label="PhsM",color="red")
# plt.scatter(df[c_names[0]],Model_resultsp_SGD,label="PhsM SGD",color="orange")
# plt.scatter(df[c_names[0]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df_sub[c_names[0]],Model_resultsp_sub,label="PhsM Sub",color="pink")
plt.scatter(df_tra[c_names[0]],Model_resultsp_tra,label="PhsM Trans",color="purple")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[1]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[1]],Model_resultsp,label="PhsM",color="red")
# plt.scatter(df[c_names[1]],Model_resultsp_SGD,label="PhsM SGD",color="orange")
# plt.scatter(df[c_names[1]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df_sub[c_names[1]],Model_resultsp_sub,label="PhsM Sub",color="pink")
plt.scatter(df_tra[c_names[1]],Model_resultsp_tra,label="PhsM Trans",color="purple")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[2]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[2]],Model_resultsp,label="PhsM",color="red")
# plt.scatter(df[c_names[2]],Model_resultsp_SGD,label="PhsM SGD",color="orange")
# plt.scatter(df[c_names[2]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df_sub[c_names[2]],Model_resultsp_sub,label="PhsM Sub",color="pink")
plt.scatter(df_tra[c_names[2]],Model_resultsp_tra,label="PhsM Trans",color="purple")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()

############################################################

plt.scatter(df_sub[c_names[0]],y_sub,label="Real",color=Real_color)
plt.scatter(df_sub[c_names[0]],Model_resultsp_sub,label="PhsM Sub",color="pink")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df_sub[c_names[1]],y_sub,label="Real",color=Real_color)
plt.scatter(df_sub[c_names[1]],Model_resultsp_sub,label="PhsM Sub",color="pink")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df_sub[c_names[2]],y_sub,label="Real",color=Real_color)
plt.scatter(df_sub[c_names[2]],Model_resultsp_sub,label="PhsM Sub",color="pink")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()

############################################################

plt.scatter(df_tra[c_names[0]],y_tra,label="Real",color=Real_color)
plt.scatter(df_tra[c_names[0]],Model_resultsp_tra,label="PhsM Tran",color="purple")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df_tra[c_names[1]],y_tra,label="Real",color=Real_color)
plt.scatter(df_tra[c_names[1]],Model_resultsp_tra,label="PhsM Tran",color="purple")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df_tra[c_names[2]],y_tra,label="Real",color=Real_color)
plt.scatter(df_tra[c_names[2]],Model_resultsp_tra,label="PhsM Tran",color="purple")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()


# In[ ]:


# Feature importance by regime
#  Identify largest coefficient magnitudes in each regime
print("Scaled Physics-Based Subsonic Intercept = {}".format(interceptp_sub))
print("Scaled Physics-Based Subsonic Coefficients: {}".format(coefficientsp_sub))
print("Physics-Based Subsonic Intercept = {}".format(true_interceptp_sub))
print("Physics-Based Subsonic Coefficients: {}".format(true_coefficientsp_sub))

print("Scaled Physics-Based Transonic Intercept = {}".format(interceptp_tra))
print("Scaled Physics-Based Transonic Coefficients: {}".format(coefficientsp_tra))
print("Physics-Based Transonic Intercept = {}".format(true_interceptp_tra))
print("Physics-Based Transonic Coefficients: {}".format(true_coefficientsp_tra))

#  Compare dominant features between regimes
df_X_phys_sub = pd.DataFrame(X_phys_sub_scaled_train)
co_mtx = df_X_phys_sub.corr(numeric_only=True)
print(co_mtx)
sns.heatmap(co_mtx,cmap="YlGnBu", annot=True)
plt.show()

print("VIF of Physics-Based Subsonic Model Features Data Frame:")
X = add_constant(df_X_phys_sub)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

df_X_phys_tra = pd.DataFrame(X_phys_tra_scaled_train)
co_mtx = df_X_phys_tra.corr(numeric_only=True)
print(co_mtx)
sns.heatmap(co_mtx,cmap="YlGnBu", annot=True)
plt.show()

print("VIF of Physics-Based Transonic Model Features Data Frame:")
X = add_constant(df_X_phys_tra)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# In[ ]:


# Cross-validation stability
#  Perform 5-fold cross-validation
n_splits_sub = 5
kf_sub = KFold(n_splits = n_splits_sub, random_state=42, shuffle=True)
kf_sub.get_n_splits()

n_splits_tra = 5
kf_tra = KFold(n_splits = n_splits_tra, random_state=42, shuffle=True)
kf_tra.get_n_splits()


# In[ ]:


#  Track coefficient values across folds
print("Subsonic KFold Process:\n")
KFold_sub = {}
for i, (train_index, test_index) in enumerate(kf_sub.split(X_phys_sub)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    # print(type(train_index))
    # print(type(X_phys_sub))
    # print(type(y_sub))
    # X_trainp_subK, X_testp_subK, y_trainp_subK, y_testp_subK = [[],[],[],[]]
    # for j in train_index:
    #     X_trainp_subK.append(X_phys_sub[j])
    #     y_trainp_subK.append(y_sub[j])
    # for j in test_index:
    #     X_testp_subK.append(X_phys_sub[j])
    #     y_testp_subK.append(y_sub[j])
    X_trainp_subK = X_phys_sub[train_index]
    X_testp_subK = X_phys_sub[test_index]
    y_trainp_subK = y_sub[train_index]
    y_testp_subK = y_sub[test_index]

    scaler_phys_subK = StandardScaler()
    X_phys_sub_scaled_trainK = scaler_phys_subK.fit_transform(X_trainp_subK)
    X_phys_sub_scaled_testK = scaler_phys_subK.transform(X_testp_subK)
    mean_subK = scaler_phys_subK.mean_
    SD_subK = scaler_phys_subK.scale_

    modelp_subK = LinearRegression()
    modelp_subK.fit(X_phys_sub_scaled_trainK, y_trainp_subK)

    y_predp_subK = modelp_subK.predict(X_phys_sub_scaled_testK)

    interceptp_subK = modelp_subK.intercept_
    coefficientsp_subK = modelp_subK.coef_

    # coefficients_nump_subK = len(coefficientsp_subK)
    true_coefficientsp_subK = coefficientsp_subK / SD_subK
    true_interceptp_subK = interceptp_subK - np.sum((coefficientsp_subK * mean_subK) / SD_subK)
    KFold_sub[str(i)] = [[interceptp_subK] + list(coefficientsp_subK), [true_interceptp_subK] + list(true_coefficientsp_subK)]
    # Model_resultsp_subK = (X_phys_sub @ true_coefficientsp_subK) + true_interceptp_subK
    print("KFold {} Unscaled Subsonic Coefficients: {}".format(i,KFold_sub[str(i)][1]))
    # print("KFold {} Subsonic Coefficients: {}, {}".format(i,true_interceptp_subK,true_coefficientsp_subK))
# print(KFold_sub.keys())

# for i, (train_index, test_index) in enumerate(kf_sub.split(X_phys_tra)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")

#  Compute mean and std of each coefficient
coefvals_sub_list_un = [[] for i in range(coefficients_nump_sub + 1)]
for i in range(coefficients_nump_sub + 1):
    # print(i)
    for j in range(n_splits_sub):
        coefvals_sub_list_un[i].append(KFold_sub[str(j)][1][i])
means_sub_un = [np.mean(coefvals_sub_list_un[i]) for i in range(coefficients_nump_sub + 1)]
SDs_sub_un = [np.std(coefvals_sub_list_un[i]) for i in range(coefficients_nump_sub + 1)]
for i in range(coefficients_nump_sub + 1):
    print("Unscaled Coefficient {}: mean = {}, standard deviation = {}".format(i,means_sub_un[i],SDs_sub_un[i]))

coefvals_sub_list = [[] for i in range(coefficients_nump_sub + 1)]
for i in range(coefficients_nump_sub + 1):
    # print(i)
    for j in range(n_splits_sub):
        coefvals_sub_list[i].append(KFold_sub[str(j)][0][i])
means_sub = [np.mean(coefvals_sub_list[i]) for i in range(coefficients_nump_sub + 1)]
SDs_sub = [np.std(coefvals_sub_list[i]) for i in range(coefficients_nump_sub + 1)]
for i in range(coefficients_nump_sub + 1):
    print("Scaled Coefficient {}: mean = {}, standard deviation = {}".format(i,means_sub[i],SDs_sub[i]))

CV_score_sub = cross_val_score(modelp_sub, X_phys_sub_scaled_train, y_trainp_sub, cv=n_splits_sub)
print("CV Scores: {}".format(CV_score_sub))
print("CV Score: {}".format(np.mean(CV_score_sub)))


# In[ ]:


#  Track coefficient values across folds
print("Transonic KFold Process:\n")
KFold_tra = {}
for i, (train_index, test_index) in enumerate(kf_tra.split(X_phys_tra)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    X_trainp_traK = X_phys_tra[train_index]
    X_testp_traK = X_phys_tra[test_index]
    y_trainp_traK = y_tra[train_index]
    y_testp_traK = y_tra[test_index]

    scaler_phys_traK = StandardScaler()
    X_phys_tra_scaled_trainK = scaler_phys_traK.fit_transform(X_trainp_traK)
    X_phys_tra_scaled_testK = scaler_phys_traK.transform(X_testp_traK)
    mean_traK = scaler_phys_traK.mean_
    SD_traK = scaler_phys_traK.scale_

    modelp_traK = LinearRegression()
    modelp_traK.fit(X_phys_tra_scaled_trainK, y_trainp_traK)

    y_predp_traK = modelp_traK.predict(X_phys_tra_scaled_testK)

    interceptp_traK = modelp_traK.intercept_
    coefficientsp_traK = modelp_traK.coef_

    true_coefficientsp_traK = coefficientsp_traK / SD_traK
    true_interceptp_traK = interceptp_traK - np.sum((coefficientsp_traK * mean_traK) / SD_traK)
    KFold_tra[str(i)] = [[interceptp_traK] + list(coefficientsp_traK), [true_interceptp_traK] + list(true_coefficientsp_traK)]
    print("KFold {} Unscaled Subsonic Coefficients: {}".format(i,KFold_tra[str(i)][1]))


#  Compute mean and std of each coefficient
coefvals_tra_list_un = [[] for i in range(coefficients_nump_tra + 1)]
for i in range(coefficients_nump_tra + 1):
    # print(i)
    for j in range(n_splits_tra):
        coefvals_tra_list_un[i].append(KFold_tra[str(j)][1][i])
means_tra_un = [np.mean(coefvals_tra_list_un[i]) for i in range(coefficients_nump_tra + 1)]
SDs_tra_un = [np.std(coefvals_tra_list_un[i]) for i in range(coefficients_nump_tra + 1)]
for i in range(coefficients_nump_tra + 1):
    print("Unscaled Coefficient {}: mean = {}, standard deviation = {}".format(i,means_tra_un[i],SDs_tra_un[i]))

coefvals_tra_list = [[] for i in range(coefficients_nump_tra + 1)]
for i in range(coefficients_nump_tra + 1):
    # print(i)
    for j in range(n_splits_tra):
        coefvals_tra_list[i].append(KFold_tra[str(j)][0][i])
means_tra = [np.mean(coefvals_tra_list[i]) for i in range(coefficients_nump_tra + 1)]
SDs_tra = [np.std(coefvals_tra_list[i]) for i in range(coefficients_nump_tra + 1)]
for i in range(coefficients_nump_tra + 1):
    print("Scaled Coefficient {}: mean = {}, standard deviation = {}".format(i,means_tra[i],SDs_tra[i]))


# scaler_phys_tra_2 = StandardScaler()
# X_phys_tra_scaled = scaler_phys_tra_2.fit_transform(X_phys_tra)
# CV_score_tra = cross_val_score(modelp_tra, X_phys_tra_scaled, y_tra, cv=n_splits_tra)
# print("CV Scores: {}".format(CV_score_tra))
# print("CV Score: {}".format(np.mean(CV_score_tra)))

CV_score_tra = cross_val_score(modelp_tra, X_phys_tra_scaled_train, y_trainp_tra, cv=n_splits_tra)
print("CV Scores: {}".format(CV_score_tra))
print("CV Score: {}".format(np.mean(CV_score_tra)))


# ### Analysis: Regime-Dependent Behavior
# 
# ** Write 2-3 paragraphs discussing:**
# - Performance comparison: global vs regime-specific models
# - Which physical effects dominate in each regime?
# - Does this align with aerodynamic theory?
# - Coefficient stability across cross-validation folds
# 
# The result of breaking the regions up into subsonic and transonic seems to have been worse than including the two regions together. My primary reason for concluding this is that the RMSE has increased and the R^2 has slightly decreased. However, this is likely just the result of having too few points in each regime, missing some subsonic effects and missing coefficient scaling produced by combining the two regimes. All the Mach Number terms dominate in both regimes; however, the more exponential terms dominate in the transonic regime. It’s also interesting to note that in both regimes the linear Mach Number term is negative. This doesn’t physically make sense but is likely caused by the effect of there being many Mach Number terms in the model.
# 
# The overall scattering of points between the global and divided models looks generally the same though. This supports that it is possible to break up the subsonic and transonic flight regimes to obtain superior fits if enough data is available. The CV Score also supports this as it is decently high. However, due to having too few points in each regime, the CV Score is lower than it should be and would benefit greatly from using the global model in this scenario.

# ## Part 2e: Heteroscedasticity and Weighted Least Squares
# 
# **Tasks:**
# 1. Diagnose heteroscedasticity formally
# 2. Implement weighted least squares
# 3. Explain weight selection
# 4. Compare WLS vs OLS

# In[ ]:


# Heteroscedasticity diagnosis
#  Plot residual variance in different Mach bins
residualsp_sub = y_testp_sub - y_predp_sub
residualsp_tra = y_testp_tra - y_predp_tra

plt.scatter(y_predp_sub, residualsp_sub, label="Subsonic")
plt.scatter(y_predp_tra, residualsp_tra, label="Transonic")
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted C_D Values')
plt.ylabel('Residuals')
plt.title('Residual vs. Fitted Values Plot')
plt.grid()
plt.legend()
plt.show()

plt.scatter(X_phys_sub_scaled_test[:,2], residualsp_sub, label="Subsonic")
plt.scatter(X_phys_tra_scaled_test[:,2], residualsp_tra, label="Transonic")
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Scaled Mach Number')
plt.ylabel('Residuals')
plt.title('Residual vs. Scaled Mach Number Plot')
plt.grid()
plt.legend()
plt.show()


#  Check if variance increases with Mach
print("Through visual inspection variance seems to increase as Mach Number increases in both the subsonic and transonic models")


# In[ ]:


# Weighted Least Squares
#  Estimate variance as function of Mach
#  Construct weights w_i = 1/sigma_i^2
# W_sub = np.zeros((n_sub,n_sub))
# for i in range(n_sub):
#     W_sub[i,i] = 1/(np.std(X_phys_sub[:,i])**2)
# print(W_sub)

# print(np.shape(X_phys_sub),np.shape(W_sub))
# beta_sub = np.matmul(X_phys_sub.T,W_sub)
# print(beta_sub)
# beta_sub = np.linalg.inv(np.matmul(beta_sub,X_phys_sub))
# print(np.shape(beta_sub))
# print(beta_sub)
# beta_sub = np.matmul(np.matmul(np.matmul(beta_sub,X_phys_sub),W_sub),y_sub)
# print(beta_sub)

# print(len(X_trainp))
# print(len(X_phys_scaled_train))
y_predp_t = modelp.predict(X_phys_scaled_train)
# print(len(y_predp_t))
# print(len(y_testp))
residualsp_t = y_trainp - y_predp_t
# print(indices_phys)
bins = np.linspace(min(X_phys[train_indicesp, 2]), max(X_phys[train_indicesp, 2]), 10)
SD_t = np.zeros_like(residualsp_t)
for i in range(len(bins)-1):
    ind = (X_phys[train_indicesp, 2] >= bins[i]) & (X_phys[train_indicesp, 2] < bins[i+1])
    SD_t[ind] = np.var(residualsp_t[ind])

for i in range(len(SD_t)): # had to do this otherwise weights would be infinity
    if SD_t[i] == 0:
        SD_t[i] = 10**-8
# print(len(SD_t))
# print(SD_t)
W = 1 / (SD_t**2)
print(W[:10])

#  Train WLS model using sample_weight parameter
modelp_WLS = LinearRegression()
modelp_WLS.fit(X_phys_scaled_train, y_trainp, sample_weight = W)
y_predp_WLS = modelp_WLS.predict(X_phys_scaled_test)
# mean_phys_WLS = modelp_WLS.mean_
# SD_phys_WLS = modelp_WLS.scale_

#  Compare WLS vs OLS performance
interceptp_t = modelp_WLS.intercept_
coefficientsp_t = modelp_WLS.coef_
coefficients_nump_t = len(coefficientsp_t)

n_phys_t = len(y_testp)
msep_t = mean_squared_error(y_testp, y_predp_WLS)
rmsep_t = np.sqrt(msep_t)
r2p_t = r2_score(y_testp, y_predp_WLS)
r2ap_t = 1 - ((1-r2p_t)*(n_phys_t-1)/(n_phys_t - (coefficients_nump_t + 1) - 1)) # plus 1 for d to account for intercept
maep_t = mean_absolute_error(y_testp, y_predp_WLS)
mapep_t = mean_absolute_percentage_error(y_testp, y_predp_WLS)
maxep_t = max_error(y_testp, y_predp_WLS)

print("Key Metrics: MSE,                  RMSE,                  R^2,                R_adj^2,            MAE,                   MAPE,                 Max Error")
print("PB:          {}, {}, {}, {}, {}, {}, {}".format(msep,rmsep,r2p,r2ap,maep,mapep,maxep))
print("PB WLS:      {}, {}, {}, {}, {}, {}, {}".format(msep_t,rmsep_t,r2p_t,r2ap_t,maep_t,mapep_t,maxep_t))

true_coefficientsp_t = coefficientsp_t / SD_phys
true_interceptp_t = interceptp_t - np.sum((coefficientsp_t * mean_phys) / SD_phys)
Model_resultsp_t = (X_phys @ true_coefficientsp_t) + true_interceptp_t


# In[ ]:


Real_color = "green" # (np.random.random(), np.random.random(), np.random.random())
plt.scatter(df[c_names[0]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[0]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[0]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[0]],Model_resultsp_t,label="PhsM WLS",color="orange")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[1]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[1]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[1]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[1]],Model_resultsp_t,label="PhsM WLS",color="orange")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[2]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[2]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[2]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[2]],Model_resultsp_t,label="PhsM WLS",color="orange")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()


# ### Analysis: Weighted Least Squares
# 
# ** Write 1-2 paragraphs explaining:**
# - How you estimated Mach-dependent variance
# - Why inverse variance weighting makes theoretical sense
# - Impact of WLS on model performance and residuals
# 
# I estimated the Mach-dependent variance by using a binning method to then produce the weights. Inverse variance weighting makes sense as values farther away from the standard deviation are statistically more likely to be outliers and thus should have less of an impact on the data. The residuals stayed roughly the same. My best guess is that I implemented WLS incorrectly or WLS is unlikely to improve results because of the significant amount of noise in the data. With that in mind it makes sense in this scenario to focus on improving the functions used to create the model before implementing WLS. 

# ## Part 2f: Outlier Analysis and Treatment
# 
# **Tasks:**
# 1. Calculate leverage and Cook's distance
# 2. Identify influential points
# 3. Physically justify outlier treatment
# 4. Sensitivity analysis with/without outliers

# In[ ]:


# Leverage calculation
#  Compute leverage values (diagonal of hat matrix)
n_phys_train = len(X_phys_scaled_train[:,0])
# H = np.zeros(n_phys_train)
mat1,mat2 = [X_phys_scaled_train.T, X_phys_scaled_train]
val = np.linalg.inv(np.matmul(mat1, mat2))
# print(np.shape(val))
# print(np.shape(mat1))
H = np.diag(np.matmul(np.matmul(mat2,val),mat1))
# for i in range(n_phys_train):
#     H[i] = np.matmul(np.matmul(mat1[:,i],val),mat2[i,:])
print(H[:20])

#  Identify high-leverage points (h_i > 2p/n)
p = 1 + len(coefficientsp)
print(p)
h_avg = p / n_phys_train
print(h_avg)
influential_ind = []
for i in range(n_phys_train):
    if H[i] > (2*h_avg):
        influential_ind.append(i)
        print("index = {}: h-val = {}".format(i,H[i]))


# In[ ]:


# Cook's distance
#  Use OLSInfluence to compute Cook's distance
phys_sm_model = sm.OLS(y_trainp,X_phys_scaled_train)
phys_results = phys_sm_model.fit()
phys_OLSinf = phys_results.get_influence()
leverage = phys_OLSinf.hat_matrix_diag
cooks_d = phys_OLSinf.cooks_distance[0]
p_values = phys_OLSinf.cooks_distance[1]
print("First 10 values of Influence: {}".format(leverage[:10]))
print("First 10 values of Cook's Distance: {}".format(cooks_d[:10]))
print("First 10 values of P-values: {}".format(p_values[:10]))
# print(cooks_d)
# print(p_values)

#  Identify influential points (D_i > 1 or D_i > 4/n)
influential_D_ind = []
for i in range(n_phys_train):
    if cooks_d[i] > (4/n_phys_train):
        influential_D_ind.append(i)
        print("index = {}: Cook's Distance = {}".format(i,cooks_d[i]))

#  Plot Cook's distance
plt.scatter(leverage,cooks_d)
plt.xlabel("Leverage (h^)")
plt.xlabel("Cook's Distance (D_i)")
plt.xlabel("Cook's Distance vs. Leverage")
plt.show()


# In[ ]:


# Examine outliers
#  Display outlier Mach numbers and characteristics
plt.scatter(X_phys_scaled_train[:,2],leverage,label="Leverage",color="blue")
plt.scatter(X_phys_scaled_train[:,2],cooks_d,label="Cook's Distance",color="red")
plt.scatter(X_phys_scaled_train[influential_D_ind,2],leverage[influential_D_ind],label="Leverage Outliers",color="green")
plt.scatter(X_phys_scaled_train[influential_D_ind,2],cooks_d[influential_D_ind],label="Cook's Distance Outliers",color="pink")
plt.xlabel("Scaled Mach Number")
plt.xlabel("Outlier Statistics")
plt.xlabel("Outlier Statistics vs. Scaled Mach Number")
plt.legend()
plt.show()

plt.scatter(X_phys_scaled_train[:,4],leverage,label="Leverage",color="blue")
plt.scatter(X_phys_scaled_train[:,4],cooks_d,label="Cook's Distance",color="red")
plt.scatter(X_phys_scaled_train[influential_D_ind,4],leverage[influential_D_ind],label="Leverage Outliers",color="green")
plt.scatter(X_phys_scaled_train[influential_D_ind,4],cooks_d[influential_D_ind],label="Cook's Distance Outliers",color="pink")
plt.xlabel("Scaled log(Re)")
plt.xlabel("Outlier Statistics")
plt.xlabel("Outlier Statistics vs. Scaled Mach Number")
plt.legend()
plt.show()


# In[ ]:


# Sensitivity analysis
#  Retrain model with outliers removed
#  Compare performance and coefficients
modelp_C = LinearRegression()
print(influential_D_ind)
# print(len(X_phys_scaled_train[:,0]))
# print(len(y_trainp))
X_phys_scaled_train_C = np.delete(X_phys_scaled_train, influential_D_ind, axis=0)
# print(len(X_phys_scaled_train_C[:,0]))
y_trainp_C = np.delete(y_trainp,influential_D_ind, axis=0)
# print(len(y_trainp_C))
modelp_C.fit(X_phys_scaled_train_C, y_trainp_C)
y_predp_C = modelp_C.predict(X_phys_scaled_test)

interceptp_C = modelp_C.intercept_
coefficientsp_C = modelp_C.coef_
coefficients_nump_C = len(coefficientsp_C)

n_phys_C = len(y_testp)
msep_C = mean_squared_error(y_testp, y_predp_C)
rmsep_C = np.sqrt(msep_C)
r2p_C = r2_score(y_testp, y_predp_C)
r2ap_C = 1 - ((1-r2p_C)*(n_phys_C-1)/(n_phys_C - (coefficients_nump_C + 1) - 1)) # plus 1 for d to account for intercept
maep_C = mean_absolute_error(y_testp, y_predp_C)
mapep_C = mean_absolute_percentage_error(y_testp, y_predp_C)
maxep_C = max_error(y_testp, y_predp_C)

print("Key Metrics: MSE,                  RMSE,                  R^2,                R_adj^2,            MAE,                   MAPE,                 Max Error")
print("PB:          {}, {}, {}, {}, {}, {}, {}".format(msep,rmsep,r2p,r2ap,maep,mapep,maxep))
print("PB Cook's:   {}, {}, {}, {}, {}, {}, {}".format(msep_C,rmsep_C,r2p_C,r2ap_C,maep_C,mapep_C,maxep_C))

true_coefficientsp_C = coefficientsp_C / SD_phys
true_interceptp_C = interceptp_C - np.sum((coefficientsp_C * mean_phys) / SD_phys)
Model_resultsp_C = (X_phys @ true_coefficientsp_C) + true_interceptp_C


# In[ ]:


Real_color = "green" # (np.random.random(), np.random.random(), np.random.random())
plt.scatter(df[c_names[0]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[0]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[0]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[0]],Model_resultsp_t,label="PhsM WLS",color="orange")
plt.scatter(df[c_names[0]],Model_resultsp_C,label="PhsM Cook's",color="purple")
plt.xlabel("alpha")
plt.ylabel("C_D")
plt.title("C_D vs. alpha")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[1]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[1]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[1]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[1]],Model_resultsp_t,label="PhsM WLS",color="orange")
plt.scatter(df[c_names[1]],Model_resultsp_C,label="PhsM Cook's",color="purple")
plt.plot([0.7,0.7],[summary_statistics[c_names[-1]]["minimum"],summary_statistics[c_names[-1]]["maximum"]],label="Mach Number Transition to Transonic Region",color="black")
plt.xlabel("Mach")
plt.ylabel("C_D")
plt.title("C_D vs. Mach")
plt.grid()
plt.legend()
plt.show()

plt.scatter(df[c_names[2]],df[c_names[-1]],label="Real",color=Real_color)
plt.scatter(df[c_names[2]],Model_resultsg,label="GenP",color="blue")
plt.scatter(df[c_names[2]],Model_resultsp,label="PhsM",color="red")
plt.scatter(df[c_names[2]],Model_resultsp_t,label="PhsM WLS",color="orange")
plt.scatter(df[c_names[2]],Model_resultsp_C,label="PhsM Cook's",color="purple")
plt.xlabel("Reynolds")
plt.ylabel("C_D")
plt.title("C_D vs. Reynolds")
plt.grid()
plt.legend()
plt.show()


# ### Analysis: Outlier Treatment
# 
# ** Write 2 paragraphs addressing:**
# 
# **Physical Justification:**
# - Where do outliers occur (Mach numbers)?
# - Recall: data generator includes outliers at M ≈ 0.7 for shock buffet
# - Should they be removed (errors), retained (real phenomena), or modeled separately?
# - Aerospace engineering reasoning for your decision
# 
# **Sensitivity:**
# - How much do outliers affect model performance and coefficients?
# - Final recommendation for outlier handling
# 
# Outliers are most likely to occur at low or high Mach Numbers with little impact from the Reynold’s number besides one or two points at the extremes. I believe the two values at the beginning of the transonic region should be remove from the model unless there is a dedicated function to capture that sharp spike included in the model. If these shock buffet points are included they will strongly throw off the model in both regimes unnecessarily. Statistically they raise the upper end of the curve leading to the overprediction in the transonic region and underprediction in the subsonic regime. All of this depends on where the mean of the C_D is though. Since more points are generally on the left of the shock buffet the effect as described will occur.
# 
# Outliers negatively impact model performance as they lead to unnecessary changes in the overall model that begin to negatively affect the predictions. If a data point can reliably, from a statistical standpoint, be removed, then it should as it can greatly improve the model if done effectively. However, this can cause safety concerns as sometimes a unique event that the model designer is missing may be carelessly thrown out. For example, the shock buffet is important to have knowledge of and presenting someone the data without it may raise concerns. My recommendation for outlier handling is to use Cook’s method because I feel like it adequately and reliably captures outliers. Some discretion needs to be used and the general tolerance for the points to be removed should be controlled by the maker of the model but in general it’s a good method. 
# 

# ---
# # Summary and Conclusions
# 
# ** Write a brief summary (1-2 paragraphs) of your key findings:**
# - Most important insights about drag prediction
# - Which modeling approach you recommend and why
# - Limitations of your analysis
# - Suggestions for future work
# 
# There are many unique effects that play into predicting drag. I learned it’s not necessarily sufficient to just use hundreds of different equations to fit a set of data. The creation of a good model requires knowledge of the important physical effects. I would recommend the physics-based model approach as it better predicted the trends of the data than using a polynomial model. Additionally, if the creator of a model has a strong understanding of the physical effects, then other parameters that affect the real-world scenario can be discovered. If a physics-based model fits well, like with the coefficient of drag data, many other parameters can be solved for, with some small degree of error, using the findings of the model.
# 
# The primary limitations of my analysis were the time it took me to learn the material. If I didn’t take so long to complete the assignment, then perhaps I could have generated a better model with better physics-based equations. The goal for future work would be to research more of the key effects involved in subsonic and transonic flight and to include them in the model. Additionally, troubleshooting the code and streamlining the processes would also be desirable.
# 
