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

warnings.filterwarnings('ignore')

#load data
dataset = pd.read_csv("C:/Users/ayhan/Desktop/ML_Project/diabetic_data.csv")

#value sayısını kontrol eden function
def NumberofValue(column, value):
    count = 0
    for i in (dataset[column]):
    
        if i == value:
            count = count + 1
   # print(count)    

#print(dataset)
#drop 'encounter_id','patient_nbr' column
dataset.drop(["encounter_id","patient_nbr"],axis=1,inplace=True)


#Race'i belli olmayan hastaları sildik.
dataset = dataset.loc[dataset["race"] != "?"]
NumberofValue("race", "?")


#Genderdan Unknown/Invalid değerini temizledim.
dataset = dataset.loc[dataset["gender"] != "Unknown/Invalid"]
NumberofValue("gender", "Unknown/Invalid")


#drop %90 dan fazlası missing olan columnları
dataset = dataset.drop(["weight","payer_code","medical_specialty"], axis = 1)


#sadece aynı değerler olduğu için bu iki columu siliyorum.
dataset = dataset.drop(["examide","citoglipton"], axis = 1)


#sadece 1. diagnosisi kullanıcağım için diag_1 ve diag_2 siliyorum.
dataset = dataset.drop(["diag_2","diag_3"], axis = 1)

#nan değere çeviriyoruz.
dataset.replace('?', np.nan, inplace = True)

#tüm nan değerleri drop ettik.
dataset= dataset.dropna()

#age cleanup
#print(dataset["age"].value_counts())

cleanup_age = {"age": {"[0-10)":5, "[10-20)":15, "[20-30)":25, "[30-40)":35, "[40-50)":45, 
                      "[50-60)":55, "[60-70)":65, "[70-80)":75, "[80-90)":85, "[90-100)":95}}


dataset.replace(cleanup_age, inplace=True)

#print(dataset["age"].value_counts())


#cleanup admission_type_id

#print(dataset["admission_type_id"].unique())
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

#print(dataset["discharge_disposition_id"].value_counts())

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

#one hot encoding
dataset = pd.get_dummies(dataset,columns = [dataset.columns.values[i] 
                                            for i in range(17,38) ], prefix=[dataset.columns.values[i] 
                                            for i in range(17,38)], prefix_sep='_',drop_first=True) 



#DATA ENCODING

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


#num lab procedures eksik veriler vardı onlardan kurtuldum
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


#undersampler yapıyoruz.
print(X.shape, y.shape)
ros = RandomUnderSampler(random_state=0)
X, y = ros.fit_resample(X, y)
print(X.shape, y.shape)
print(" ")
#train split yapıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("\nRFC Classifier")
#random forest classifier
RFC = RandomForestClassifier(max_depth=6).fit(X_train, y_train)
RFC_pred = RFC.predict(X_test)

print("{} Accuracy: %".format("ROC"), roc_auc_score(y_test, RFC_pred) * 100)

print("\nKNN Classifier")
#KNN classifer

KNN = KNeighborsClassifier().fit(X_train, y_train)
KNN_pred = KNN.predict(X_test)

print("{} Accuracy: %".format("ROC"), roc_auc_score(y_test, KNN_pred) * 100)


#GBC classifier
print("\nGBC classifier")

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
    max_depth=7, random_state=0).fit(X_train, y_train)
GBC_pred = GBC.predict(X_test)

print("{} Accuracy: %".format("ROC"), roc_auc_score(y_test, GBC_pred) * 100)


#Logistic regression
print("\nlogisticRegression")

LOG=LogisticRegression().fit(X_train, y_train)
LOG_pred = LOG.predict(X_test)

print("{} Accuracy: %".format("ROC"), roc_auc_score(y_test, LOG_pred) * 100)
















