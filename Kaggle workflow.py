
# coding: utf-8

# In[63]:


# The aim of this project is to create a data science workflow. Its geared towards
# Kaggle competitions, however, it can be used as a template for other projects as well. 
# In this, I will use the Titanic competition to create the flow. 


# In[64]:


# Importing the necessary libraries 
import pandas as pd
import numpy as np 

# Reading the dataset 
train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")
print(holdout.head(5))


# In[65]:


# %load functions.py
def process_missing(df):
    """Handle various missing values from the data set

    Usage
    ------

    holdout = process_missing(holdout)
    """
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------

    train = process_age(train)
    """
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    """Process the Fare column into pre-defined 'bins' 

    Usage
    ------

    train = process_fare(train)
    """
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    """Process the Cabin column into pre-defined 'bins' 

    Usage
    ------

    train process_cabin(train)
    """
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df

def process_titles(df):
    """Extract and categorize the title from the name column 

    Usage
    ------

    train = process_titles(train)
    """
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[66]:


# Using the helper functions that just got loaded, crete a function that takes all
# into account. 
def all_fun(df):
    df = process_missing(df)
    df = process_age(df)
    df = process_fare(df)
    df = process_titles(df)
    df = process_cabin(df)
    
    for i in ["Age_categories","Fare_categories","Title","Cabin_type","Sex"]:
        df = create_dummies(df,i)
    return df 

train = all_fun(train)
holdout = all_fun(holdout)


# In[67]:


print(train.head(5))


# In[68]:


# The flow will be : Exploring data->Feature Engineering->Feature Selection->Model Training
# Exploring the "SibSp" and "Parch" columns 
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')
features = ["SibSp","Parch"]
fig = plt.figure()
ax = plt.axes() 
ax.hist(train["SibSp"], bins=8,alpha = 0.5,color = "green")

ax.hist(train["Parch"], bins = 6, color="blue", alpha = 0.5)
ax.legend(loc="upper right")
plt.show()


# In[69]:


# Make a pivot table for more insights 
pivot_df = train[["SibSp","Parch","Survived"]]
for i in features:
    pivot_table = pivot_df.pivot_table(index = i, values = "Survived")
    pivot_table.plot.bar(ylim=(0,1))
    plt.show()


# In[70]:


# Looks like people with 1-2 people in their families had the highest survival rate
# To explore this more, I'll create a new column that addresses this observation


# In[71]:


# Creating a new column that has 1 if passenger alone, and 0 if passenger has 1+
# people on board. 
def new_col(df):
    df["family"] = df[["SibSp","Parch"]].sum(axis =1)
    df.loc[df["family"]==0,"isalone"] = 1
    df.loc[df["family"]==1,"isalone"] = 0
    return df

train = new_col(train)
holdout = new_col(holdout)

print(train["isalone"].head(5))


# In[72]:


# For feature selection, I will use Recursive Feature Elimination
# to get the best performing features
from sklearn.feature_selection import RFECV 
from sklearn.ensemble import RandomForestClassifier

# The function that I will use to incorporate RFECV and RandomForests
def select_features(df):
    
    # Select only numeric columns
    df= df.select_dtypes(include = [np.number])
    df = df.drop(columns = "isalone",axis=1)
    
    # Creating all_X,all_y
    all_X = df.drop(columns=["PassengerId","Survived"],axis=1)
    all_y = df["Survived"]
    
    rfc = RandomForestClassifier()
    selector = RFECV(rfc, cv=10)
    selector.fit(all_X,all_y)
    best_columns = all_X.columns[selector.support_]
    print(best_columns)
    return best_columns
best_columns = select_features(train)


# In[84]:


# For best model search, I will use GridSearchCV with KNN, Logarithmic regression
# and RandomForests to see which one works the best

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

score=dict()
# Defining the fucntion that does the tuning 
def select_model(df,features):
    all_X = df[features]
    all_y = df["Survived"] 
    
    # Defining the models 
    models = [
        {
            "name":"KNeighborsClassifier",
            "estimator": KNeighborsClassifier(),
            "hyperparameters":
            {
                "n_neighbors" : range(1,20,2),
                "weights" : ["distance","uniform"],
                "algorithm" : ["ball_tree","kd_tree","brute"],
                "p" : [1,2]
            }
        },
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
            {
            "solver": ["newton-cg","lbfgs","liblinear"]
            }
        },
        {
            "name": "RandomForestClassifier",
            "estimator":RandomForestClassifier(),
            "hyperparameters":
            {
                "n_estimators": [4,6,9],
                "criterion": ["entropy","gini"],
                "max_depth" : [2,5,10],
                "min_samples_leaf" : [1,5,8],
                "min_samples_split":[2,3,5]
            }
        }     
    ]
    
    # Iterate over the models and adding to the dictionary the relevant results
    for i in models:
        print(i["name"])
        grid = GridSearchCV(i["estimator"], param_grid = i["hyperparameters"],cv=10)
        grid.fit(all_X,all_y)
        i["best_param"] = grid.best_params_
        i["best_score"] = grid.best_score_
        i["best_estimator"] = grid.best_estimator_
    return models

best_models = select_model(train,best_columns)


# In[ ]:


# For Kaggle, its useful to create a function that will store the models in a.csv 
# everytime I want to run it. 
def save_submission_file(model,cols,filename):
    predictions = model.predict(holdout[features])
    submission = {
        "PassengerId": holdout["PassengerId"],
        "Survived" : predictions
        }
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(filename,index=False)

