# import
import os
import pandas  as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ##########################
# STEP 1 : READING CSV FILES
# ##########################

# Get the list of all files and directories
path = "C:\\TRASH\\datasets\\AirbnbEurope"
dir_list = os.listdir(path)

# ----------------------------------------------
# mergeCSVFile : path of CSV files
# Read all CSV file in the dataset directory
# Store data in a dataframe
# Add 2 columns for City and Week Time
# Return the dataframe with all merged values
# ----------------------------------------------
def mergeCSVFile(path):
    dfMergedCSV = pd.DataFrame()
    dfReadCSV = pd.DataFrame()
    # For each CSV file
    for csvFile in os.listdir(path):
        # Read
        dfReadCSV = pd.read_csv(path+'\\'+csvFile)
        # Look for City name and week time
        posCity = csvFile.find('_')
        posWeekTime = csvFile.find('.')
        # Add City and Week time
        dfReadCSV['city'] = csvFile[0:posCity]
        dfReadCSV['weektime'] = csvFile[posCity+1:posWeekTime]
        # Append in the merged dataframe
        dfMergedCSV = pd.concat([dfMergedCSV,dfReadCSV])
    return dfMergedCSV


# Read all CSV files and store all data in a dataframe
print("\nRead CSV files")
print("**************")
dfEuropePriceAllCSVFiles = mergeCSVFile(path)

# #############################
# STEP 1 : DATA PRESENTATION
# #############################
print("\nDescribe data")
print("*************")
print(dfEuropePriceAllCSVFiles.describe())
print("\nDataset info")
print("*************")
dfEuropePriceAllCSVFiles.info()

# #############################
# STEP 2 : DATA TRANSFORMATION
# #############################

# Separate the target and independent variables
X = dfEuropePriceAllCSVFiles.copy()

# Get Dummies for identified columns
X  = pd.get_dummies(X , columns=['room_type','host_is_superhost', 'city', 'weektime'])

# ----------------------------------------------
# getColumnPosition : Get column position from a dataframe
# ----------------------------------------------
def getColumnPosition(df, columnName):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnName):
            return i

# Create Dummy for non-multi and non-biz host : private_seller
# iat needs the column position
X["private_seller"] = 0
colPosition = getColumnPosition(X, "private_seller")
for i in range(len(X)):
    if X["multi"].iloc[i]==0 and X["biz"].iloc[i]==0 :
        X.iat[i, colPosition] = 1


# Remove id column
X = X.drop('Unnamed: 0', axis=1)
# Remove room_shared and room_private as they are included in room_type
X = X.drop(['room_shared'], axis=1)
X = X.drop(['room_private'], axis=1)
# Remove Latitude and Longitude
X = X.drop(['lat'], axis=1)
X = X.drop(['lng'], axis=1)

print("\nPrint all features")
print("******************")
print(list(X.keys()))


# #####################################
# STEP 3 : HEATMAP & CORRELATION MATRIX
# #####################################

import seaborn as sns

# ----------------------------------------------
# showHeatmap : Plot heatmap for target to see
# correlation with other features
# ----------------------------------------------
def showHeatmap(X):
    # Plot the heatmap.
    heatmap = sns.heatmap(X.corr(numeric_only=True) [['realSum']].\
    sort_values(by='realSum',
    ascending=False), vmin=-1, vmax=1, annot=True, xticklabels=True, yticklabels=True)
    heatmap.set_title('Airbnb Prices',
    fontdict={'fontsize':18}, pad=16)
    heatmap.figure.tight_layout()
    plt.show()

# ----------------------------------------------
# showCorrelationMatrix : Plot the correlation matrix
# for all features and the target
# ----------------------------------------------
def showCorrelationMatrix(X):
    corr = X.corr()
    heatmap = sns.heatmap(corr, cmap="Blues", annot=False, xticklabels=True, yticklabels=True)
    heatmap.figure.tight_layout()
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 6)
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 6)
    heatmap.set_title('Correlation Matrix')
    plt.show()

print("\nGenerate heatmap")
print("****************")
showHeatmap(X)

print("\nShow Correlation matrix")
print("***********************")
showCorrelationMatrix(X)

# #######################
# STEP 4 : DEFINE TARGET
# #######################
del X['realSum']
y = dfEuropePriceAllCSVFiles['realSum']


# ##########################################
# STEP 5 : IDENTIFYING SIGNIFICANT FEATURES
# ##########################################

# STEP 5.1 : Backward Feature Elimination
# ##########################################
# ----------------------------------------------
# getRFEFeatures : Get the significant features
# using Recursive Feature Elimination
# ----------------------------------------------
def getRFEFeatures(X,y):
    # Create the object of the model
    model = LinearRegression()
    # Specify the number of features to select
    rfe = RFE(model, n_features_to_select=10)
    # fit the model
    rfe = rfe.fit(X, y)
    RFEFearures = []
    selectedFeatures = list(X.keys())
    for i in range(0, len(selectedFeatures)):
        if(rfe.support_[i]):
            #print(selectedFeatures[i])
            RFEFearures.append(selectedFeatures[i])
    print(RFEFearures)
    return RFEFearures


# STEP 5.2 : Forward Feature Selection
# ##########################################
# ----------------------------------------------
# getFFSFeatures : Get the significant features
# using Forward-Feature Selection
# ----------------------------------------------
def getFFSFeatures(X,y):
    ffs = f_regression(X, y)
    selectedFeatures = pd.DataFrame(columns=['feature', 'ffs'])
    for i in range(0, len(X.columns)):
        insert_row = {
            "feature": X.columns[i],
            "ffs": ffs[0][i]
        }
        selectedFeatures = pd.concat([selectedFeatures, pd.DataFrame([insert_row])])

    selectedFeatures = selectedFeatures.sort_values(by=['ffs'],ascending=False,ignore_index=True)
    FFEFeatures = []
    for i in range(0, 10):
        FFEFeatures.append(selectedFeatures["feature"][i])
    print(FFEFeatures)
    return FFEFeatures

# STEP 5.3 : RandomForestRegressor Important features
# ####################################################
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------------------------
# getFeatureImportance : Get the importance features
# using RandomForestRegressor model
# ----------------------------------------------
def getFeatureImportance(X,y):
    # Saving feature names for later use
    feature_list = list(X.columns)
    # Convert to numpy array
    features = np.array(X)
    labels = np.array(y)
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    # Train the model on training data
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.25, random_state=42)
    rf.fit(train_features, train_labels)
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    IFFeatures = []
    for i in range(0, 10):
        IFFeatures.append(feature_importances[i][0])
    print(IFFeatures)
    return IFFeatures

# ----------------------------------------------
# getCommonFeatures : Get the common features
# found with the 3 methods : RFE, FFS, and
# Importance features
# ----------------------------------------------
# def getCommonFeatures (f1,f2,f3):
#     a = np.array(f1)
#     b = np.array(f2)
#     c = np.array(f3)
#     return np.intersect1d(np.intersect1d(a, b), c)

# Look for top main features from different methods : RFE, FFS, Random Forest Important features
print("\nTop features from Recursive Feature Elimination")
print("******************")
RFEFeatures = getRFEFeatures(X,y)
print("\nTop features from Forward-Feature Selection")
print("******************")
FFEFeatures = getFFSFeatures(X,y)
print("\nTop features from feature importance")
print("******************")
IFFeatures = getFeatureImportance(X, y)
#Look for common features between the different methods
# print('COMMON FEATURES')
# print(getCommonFeatures(RFEFeatures, FFEFeatures, IFFeatures))


# ##########################################
# STEP 5 : MODEL FEATURES
# ##########################################
# Model 1 : Strong features
#X = X[['room_type_Entire home/apt','room_type_Private room', 'room_type_Shared room',  'person_capacity', 'bedrooms', 'dist', 'metro_dist']]
# Model 2 : Heatmap
#X = X[['attr_index_norm', 'bedrooms', 'person_capacity', 'city_amsterdam', 'attr_index', 'room_type_Entire home/apt', 'rest_index_norm', 'city_paris', 'city_london']]
# Model 3 : RFE features
#X = X[['city_amsterdam', 'city_athens', 'city_barcelona', 'city_berlin', 'city_budapest', 'city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna']]
# Model 4 : FFS features
#X = X[['attr_index_norm', 'bedrooms', 'person_capacity', 'city_amsterdam', 'attr_index', 'room_type_Entire home/apt', 'room_type_Private room', 'rest_index_norm', 'rest_index', 'city_paris']]
# Model 5 : Random Forest important features
#X = X[['attr_index_norm', 'metro_dist', 'bedrooms', 'rest_index_norm', 'person_capacity', 'dist', 'rest_index', 'guest_satisfaction_overall', 'attr_index', 'city_amsterdam']]
# Model 6 : all significant features
X = X[['person_capacity', 'multi', 'cleanliness_rating', 'guest_satisfaction_overall', 'bedrooms', 'metro_dist', 'attr_index', 'attr_index_norm', 'rest_index', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'host_is_superhost_False', 'host_is_superhost_True', 'city_amsterdam', 'city_athens', 'city_barcelona', 'city_berlin', 'city_budapest', 'city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna', 'weektime_weekdays', 'weektime_weekends', 'private_seller']]

print("\nStudied features")
print("****************")
print(list(X.keys()))


# #####################################
# STEP 5 : Simple Linear Regression
# #####################################

from sklearn.metrics import mean_squared_error


# ----------------------------------------------
# getPredictionsOLS : Get predictions
# from a Linear Regression model
# ----------------------------------------------
def getPredictionsOLS (X_train,y_train, X_test, displaySummary):
    model = sm.OLS(y_train, X_train).fit()
    if displaySummary:
        print(model.summary())
    predictions = model.predict(X_test)
    return predictions

# ----------------------------------------------
# evaluateOLSModel : Evaluate and return
# the Root Mean Square Error of the model
# ----------------------------------------------
def evaluateOLSModel(predictions, y_test):
    mse = mean_squared_error(predictions, y_test)
    rmse = np.sqrt(mse)
    print("---Linear Regression : Root mean square error: " + str(rmse))
    return rmse

import matplotlib.pyplot as plt
# ----------------------------------------------
# plotPredictionVsActual : Plot the prediction
# values against the actual ones
# ----------------------------------------------
def plotPredictionVsActual(title, y_actual, y_predicted):
    plt.scatter(y_actual, y_predicted)
    #plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--')
    plt.show()


print("\n*********************************************")
print("Linear Regression and Cross Fold Validation  ")
print("*********************************************")
# Adding an intercept
Xols = X.copy()
Xols = sm.add_constant(Xols)
from sklearn.model_selection import KFold

# prepare cross validation with three folds.
kfold = KFold(n_splits=3, shuffle= True)
rmseList = []
count = 0

for train_index, test_index in kfold.split(Xols):
    count += 1
    print("***K-fold: " + str(count))
    X_train = Xols.loc[Xols.index.isin(train_index)]
    X_test = Xols.loc[Xols.index.isin(test_index)]
    y_train = y.loc[y.index.isin(train_index)]
    y_test = y.loc[y.index.isin(test_index)]

    predictions = getPredictionsOLS(X_train, y_train, X_test, True)
    rmse = evaluateOLSModel(predictions, y_test)
    rmseList.append(rmse)
    #plotPredictionVsActual("No scalar", y_test, predictions)

print('****************************************')
print("Linear Regression: Scores for all folds:")
print("RMSE Average : " + str(np.mean(rmseList)))
print("RMSE SD: " + str(np.std(rmseList)))

# #########################################
# STEP 6 : Linear Regression with Scalars
# #########################################
print("\n*********************************************************")
print("Linear Regression with Scalar and Cross Fold Validation  ")

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Declare scalars
sc_xMinMax = MinMaxScaler()
sc_yMinMax = MinMaxScaler()
sc_xStandard = StandardScaler()
sc_yStandard = StandardScaler()
sc_xRobust = RobustScaler()
sc_yRobust = RobustScaler()

# Define list of scalars
scalers_x = [sc_xMinMax,sc_xStandard,sc_xRobust]
scalers_y = [sc_yMinMax,sc_yStandard,sc_yRobust]
scalers_name = ["MinMax", "Standard", "Robust"]
for sc_index in range(0, len(scalers_x)):

    print("****************************************")
    print(scalers_name[sc_index] + "Scores for all folds:")

    #prepare cross validation with three folds.
    kfold = KFold(n_splits=3, shuffle= True)
    rmseList = []
    count = 0
    for train_index, test_index in kfold.split(Xols):
        count += 1
        print("***K-fold: " + str(count))
        X_train = Xols.loc[Xols.index.isin(train_index)]
        X_test = Xols.loc[Xols.index.isin(test_index)]
        y_train = y.loc[y.index.isin(train_index)]
        y_test = y.loc[y.index.isin(test_index)]

        X_train_scaled = scalers_x[sc_index].fit_transform(X_train)  # Always fit scalar training data
        X_test_scaled = scalers_x[sc_index].transform(X_test)
        y_train_scaled = scalers_y[sc_index].fit_transform(np.array(y_train).reshape(-1, 1))

        unscaledPredictions = getPredictionsOLS (X_train_scaled,y_train_scaled, X_test_scaled, False)
        predictions = scalers_y[sc_index].inverse_transform(np.array(unscaledPredictions).reshape(-1,1))
        rmse = evaluateOLSModel(predictions, y_test)
        rmseList.append(rmse)
        #plotPredictionVsActual(scalers_name[sc_index], y_test, predictions)

    print(scalers_name[sc_index] + "Linear Regression with Scalar: Scores for all folds:")
    print("RMSE Average : " + str(np.mean(rmseList)))
    print("RMSE SD: " + str(np.std(rmseList)))

del Xols

# #########################
# STEP 7 : Random Forest
# ########################
print("\n**************")
print("Random Forest ")
print("**************")

# prepare cross validation with three folds.
kfold = KFold(n_splits=3, shuffle= True)
rmseList = []
count = 0

for train_index, test_index in kfold.split(X):
    X_train = X.loc[X.index.isin(train_index)]
    X_test = X.loc[X.index.isin(test_index)]
    y_train = y.loc[y.index.isin(train_index)]
    y_test = y.loc[y.index.isin(test_index)]

    # Random Forest Regressor with 3 estimators
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    count += 1
    print("***K-fold: " + str(count))
    print('RMSE:', rmse)

    rmseList.append(rmse)
    plotPredictionVsActual("Random Regressor", y_test, predictions)

print("****************************************")
print("Random Forest Regression: Scores for all folds:")
print("RMSE Average : " + str(np.mean(rmseList)))
print("RMSE SD: " + str(np.std(rmseList)))


# ################################################
# STEP 8 : Stacked model and Cross Fold Validation
# ################################################
print("\n*********************************************")
print("Stacked model and Cross Fold validation")
print("*********************************************")
from sklearn.linear_model    import ElasticNet
from sklearn.tree            import DecisionTreeRegressor
from sklearn.svm             import SVR
from sklearn.ensemble        import AdaBoostRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.ensemble        import ExtraTreesRegressor

# ----------------------------------------------
# getUnfitModels : Get a list of base models
# ----------------------------------------------
def getUnfitModels():
    models = list()
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(ExtraTreesRegressor(n_estimators=10))
    return models

# ----------------------------------------------
# evaluateModel : Evaluate and return
# the Root Mean Square Error of a model
# ----------------------------------------------
def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse),3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)

# ----------------------------------------------
# fitBaseModels : Fit base models and return
# the list of predictions from these models
# ----------------------------------------------
def fitBaseModels(X_train, y_train, X_test, y_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train.values.ravel())
        predictions = models[i].predict(X_test)
        evaluateModel(y_test, predictions, models[i])
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models

# ----------------------------------------------
# fitStackedModel : Fit stacked model and
# return the model
# ----------------------------------------------
def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# ----------------------------------------------
# fitAllModels : Fit stacked model with
# the predictions from the base models
# ----------------------------------------------
def fitAllModels(X,y):
    # Get base models.
    unfitModels = getUnfitModels()
    models       = None
    stackedModel = None
    kfold  = KFold(n_splits=3, shuffle= True)
    y = y.to_frame()
    count = 0
    for train_index, test_index in kfold.split(X):
        count += 1
        print("***K-fold: " + str(count))
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test  = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test  = y.loc[y.index.intersection(test_index), :]

        # Fit base and stacked models.
        dfPredictions, models = fitBaseModels(X_train, y_train, X_test, y_test, unfitModels)
        stackedModel          = fitStackedModel(dfPredictions, y_test)

    return models, stackedModel


# ----------------------------------------------
# evaluateBaseAndStackModelsWithUnseenData :
# Evaluate stacked model with test data
# ----------------------------------------------
def evaluateBaseAndStackModelsWithUnseenData(X,y, models, stackedModel):
    # Evaluate base models with validation data.
    print("** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("** Evaluate Stacked Model **")
    evaluateModel(y, stackedPredictions, stackedModel)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)

print("Fitting models: ")
models, stackedModel = fitAllModels(X_train, y_train)

print("\nEvaluating models with unseen data: ")
evaluateBaseAndStackModelsWithUnseenData(X_test, y_test, models, stackedModel)

print('END')


