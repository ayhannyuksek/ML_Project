# -*- coding: utf-8 -*-
"""
Created on Sun May  9 02:50:08 2021

@author: ayhan
"""
import pandas as pd
import warnings
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import seaborn as sns
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv("C:/Users/ayhan/Desktop/ML_Project/diabetic_data.csv")

# Functions for all classification algorithms.
def classifierAlgorithm(X,y):
        #we did undersampling.
    print("Shape before undersampling")
    print(X.shape, y.shape)
    ros = RandomUnderSampler(sampling_strategy=0.5,random_state=0)
    X, y = ros.fit_resample(X, y)
    print("")
    print("Shape after undersampling")
    print(X.shape, y.shape)
    print(" ")
    
    #train split yapıyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    
    
    #random forest classifier
    print("\nRFC Classifier:")
    
    RFC = RandomForestClassifier(max_depth=6).fit(X_train, y_train)
    RFC_pred = RFC.predict(X_test)
    
    print("{} : %".format("ROC"), roc_auc_score(y_test, RFC_pred) * 100)
    print("{} : ".format("f1_score"), f1_score(y_test, RFC_pred))
    print("{} : ".format("recall"), recall_score(y_test, RFC_pred))
    print("{} : ".format("precision"), precision_score(y_test, RFC_pred))
    
    
    #KNN classifer
    print("\nKNN Classifier:")
    
    
    KNN = KNeighborsClassifier().fit(X_train, y_train)
    KNN_pred = KNN.predict(X_test)
    
    print("{} S: %".format("ROC"), roc_auc_score(y_test, KNN_pred) * 100)
    print("{} : ".format("f1_score"), f1_score(y_test, KNN_pred))
    print("{} : ".format("recall"), recall_score(y_test, KNN_pred))
    print("{} : ".format("precision"), precision_score(y_test, KNN_pred))
    
    #GBC classifier
    print("\nGBC classifier:")
    
    GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1).fit(X_train, y_train)
    GBC_pred = GBC.predict(X_test)
    
    print("{} : %".format("ROC"), roc_auc_score(y_test, GBC_pred) * 100)
    print("{} : ".format("f1_score"), f1_score(y_test, GBC_pred))
    print("{} : ".format("recall"), recall_score(y_test, GBC_pred))
    print("{} : ".format("precision"), precision_score(y_test, GBC_pred))
    
    #Logistic regression
    print("\nlogisticRegression:")
    
    LOG=LogisticRegression().fit(X_train, y_train)
    LOG_pred = LOG.predict(X_test)
    
    print("{} : %".format("ROC"), roc_auc_score(y_test, LOG_pred) * 100)
    print("{} : ".format("f1_score"), f1_score(y_test, LOG_pred))
    print("{} : ".format("recall"), recall_score(y_test, LOG_pred))
    print("{} : ".format("precision"), precision_score(y_test, LOG_pred))
    
    
    #we create a some graph
    
    x=dataset['readmitted'].value_counts().values
    sns.barplot([0,1],x)
    plt.title('Readmitted variable count before undersampling')
    
    
    
    plt.figure(figsize=(14, 7))
    ax = plt.subplot(111)
    
    
    
    #we create a model comperison graph
    models = ['Logistic Regression', 'Random Forests',"KNN Classifier","G.Boosting" ]
    values = [roc_auc_score(y_test, LOG_pred) * 100,roc_auc_score(y_test, RFC_pred) * 100, roc_auc_score(y_test, KNN_pred) * 100,
             roc_auc_score(y_test, GBC_pred) * 100 ]
    model = np.arange(len(models))
    plt.title('Comparison of Learning Algorithms with Undersampling')
    plt.bar(model, values, align='center', width = 0.15, alpha=0.7, color = 'red', label= 'ROC score')
    plt.xticks(model, models)


#value sayısını kontrol eden function
def NumberofValue(column, value):
    count = 0
    for i in (dataset[column]):
    
        if i == value:
            count = count + 1
    #print(count)    

  
#drop 'encounter_id','patient_nbr' column
dataset.drop(["encounter_id","patient_nbr"],axis=1,inplace=True)


#We deleted patients whose Race was unknown.
NumberofValue("race", "?")
dataset = dataset.loc[dataset["race"] != "?"]
NumberofValue("race", "?")


#I cleared the Unknown / Invalid value from Gender.
NumberofValue("gender", "Unknown/Invalid")
dataset = dataset.loc[dataset["gender"] != "Unknown/Invalid"]
NumberofValue("gender", "Unknown/Invalid")


#I deleted the data mostly missing value.
dataset = dataset.drop(["weight","payer_code","medical_specialty"], axis = 1)


#we just dropped it because they were the same values.
dataset = dataset.drop(["examide","citoglipton"], axis = 1)


#I'm deleting diag_1 and diag_2 because I will only use diagnosis 1.
dataset = dataset.drop(["diag_2","diag_3"], axis = 1)

#We convert it to nan value.
dataset.replace('?', np.nan, inplace = True)

# We dropped all nan values.
dataset= dataset.dropna()

#age cleanup
#print(dataset["age"].value_counts())

cleanup_age = {"age": {"[0-10)":5, "[10-20)":15, "[20-30)":25, "[30-40)":35, "[40-50)":45, 
                      "[50-60)":55, "[60-70)":65, "[70-80)":75, "[80-90)":85, "[90-100)":95}}


dataset.replace(cleanup_age, inplace=True)




#cleanup admission_type_id
mapped = {1:"Emergency",
          2:"Emergency",
          3:"Elective",
          4:"New Born",
          5:np.nan,
          6:np.nan,
          7:"Trauma Center",
          8:np.nan}

dataset["admission_type_id"] = dataset["admission_type_id"].replace(mapped)
dataset= dataset.dropna()


#cleanup discharge diposition id

mapped_discharge = {1:"Discharged to Home",
                    6:"Discharged to Home",
                    8:"Discharged to Home",
                    13:"Discharged to Home",
                    19:"Discharged to Home",
                    18:np.nan,25:np.nan,26:np.nan,
                    2:"Other",3:"Other",4:"Other",
                    5:"Other",7:"Other",9:"Other",
                    10:"Other",11:"Other",12:"Other",
                    14:"Other",15:"Other",16:"Other",
                    17:"Other",20:"Other",21:"Other",
                    22:"Other",23:"Other",24:"Other",
                    27:"Other",28:"Other",29:"Other",30:"Other"}

dataset["discharge_disposition_id"] = dataset["discharge_disposition_id"].replace(mapped_discharge)
dataset= dataset.dropna()
#print(dataset["discharge_disposition_id"].unique())


#cleanup admission source id
#print(dataset["admission_source_id"].value_counts())
mapped_adm = {1:"Referral",2:"Referral",3:"Referral",
              4:"Other",5:"Other",6:"Other",10:"Other",22:"Other",25:"Other",
              9:"Other",8:"Other",14:"Other",13:"Other",11:"Other",
              15:np.nan,17:np.nan,20:np.nan,21:np.nan,
              7:"Emergency"}

dataset["admission_source_id"] = dataset["admission_source_id"].replace(mapped_adm)

#print(dataset["admission_source_id"].value_counts())
dataset= dataset.dropna()





#DATA ENCODING

#one hot encoding
dataset = pd.get_dummies(dataset,columns = [dataset.columns.values[i] 
                                            for i in range(17,38) ], prefix=[dataset.columns.values[i] 
                                            for i in range(17,38)], prefix_sep='_',drop_first=True) 

#race
#print(dataset["race"].value_counts())
mapped_race = {"Caucasian":0,"AfricanAmerican":1, "Hispanic":2, "Other":3, "Asian":4 }
dataset["race"] = dataset["race"].replace(mapped_race)



#gender

#print(dataset["gender"].value_counts())

mapped_gender = {"Male":0,"Female":1}
dataset["gender"] = dataset["gender"].replace(mapped_gender)


#admission type

#print(dataset["admission_type_id"].value_counts())

mapped_admission_id = {"Emergency":0, "Elective":1, "New Born":2, "Trauma Center":3}
dataset["admission_type_id"] = dataset["admission_type_id"].replace(mapped_admission_id)


#discharged dispoint

#print(dataset["discharge_disposition_id"].value_counts())

mapped_discharge = {"Discharged to Home":0, "Other":1}
dataset["discharge_disposition_id"] = dataset["discharge_disposition_id"].replace(mapped_discharge)


#admission source id
#print(dataset["admission_source_id"].value_counts())

mapped_source = {"Referral":0, "Emergency":1, "Other":2}
dataset["admission_source_id"] = dataset["admission_source_id"].replace(mapped_source)


#num_lab_procedures also had missing data I got rid of them.

#print(dataset["num_lab_procedures"].unique())
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["num_lab_procedures"])
dataset["num_lab_procedures"] = label_encoder.transform(dataset["num_lab_procedures"])

#num medicitaiton

#print(dataset["num_medications"].unique())
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["num_medications"])
dataset["num_medications"] = label_encoder.transform(dataset["num_medications"])
#print(dataset["num_medications"].unique())


#diag_1

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["diag_1"])
dataset["diag_1"] = label_encoder.transform(dataset["diag_1"])
#print(dataset["diag_1"].unique())



#max_glu_serum   
  
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["max_glu_serum"])
dataset["max_glu_serum"] = label_encoder.transform(dataset["max_glu_serum"])
#print(dataset["max_glu_serum"].unique())

#a1cresult

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["A1Cresult"])
dataset["A1Cresult"] = label_encoder.transform(dataset["A1Cresult"])
#print(dataset["A1Cresult"].unique())

#change

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["change"])
dataset["change"] = label_encoder.transform(dataset["change"])
#print(dataset["change"].unique())

#diabetesmed

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset["diabetesMed"])
dataset["diabetesMed"] = label_encoder.transform(dataset["diabetesMed"])
#print(dataset["diabetesMed"].unique())

#target readmitted

mapped_readmitted = {"<30":0, ">30":1, "NO":1}
dataset["readmitted"] = dataset["readmitted"].replace(mapped_readmitted)

print("*********************************************************************************************")

#MODELİNG

#setup dataset_features
features_data=[]
for i in dataset:
    if i != "readmitted":    
        features_data.append(i)
    else:
        continue

X = dataset[features_data]
y = dataset["readmitted"] #ground truth vector

#for full data.
print("Full Data Analysis:\n")
classifierAlgorithm(X,y)


#for subsample data.
X =X.iloc[:1500,:]
y=y.iloc[:1500]
print("************************************")
print("\nSubsample Data Analysis:\n")
classifierAlgorithm(X,y)



