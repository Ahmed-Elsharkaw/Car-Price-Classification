


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statistics import mode
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
     

def func(x):
    x = x.replace('[', '')
    x = x.replace('(', '')
    x = x.replace(')', '')
    x = x.replace(']', '')
    return x

    
brandModelMode={}
CategoryMode = 'moderate'
brandMode={}
def feature_engineering(X):
    li = []
    df = pd.DataFrame(X)
    df.drop('car_id', axis='columns', inplace=True)
    new = df["car-info"].str.split(",", n=3, expand=True)
    #brandMode
    for n in range(df.shape[0]):
        current = df['car-info'][n]
        category = df['Price Category'][n]
        currentSplit = current.split(",", 3)
        if brandMode.get(currentSplit[1], -1) == -1:
            brandMode.update({currentSplit[1]: [category]})
        else:
            updatedList = brandMode.get(currentSplit[1])
            updatedList.append(category)
            brandMode.update({currentSplit[1]: updatedList})
    for item in brandMode.items():
        categories = item[1]
        key = item[0]
        wantedMode = mode(categories)
        brandMode.update({key: wantedMode})
    # brandModelMode
    for n in range(df.shape[0]):
        current = df['car-info'][n]
        category = df['Price Category'][n]
        if brandModelMode.get(current, -1) == -1:
            brandModelMode.update({current: [category]})
        else:
            updatedList = brandModelMode.get(current)
            updatedList.append(category)
            brandModelMode.update({current: updatedList})
    for item in brandModelMode.items():
        categories = item[1]
        key = item[0]
        wantedMode = mode(categories)
        brandModelMode.update({key: wantedMode})
    for n in range(df.shape[0]):
        str = df['car-info'][n]
        li.append(brandModelMode.get(str))
    df['mode category'] = li
    df['mode category'].replace({"cheap": 0, "moderate": 1, "expensive": 2, "very expensive": 3}, inplace=True)
    df['model'] = new[0].apply(func)
    df['brand'] = new[1].apply(func)
    df['year'] = new[2].apply(func)
    df['year'] = df['year'].apply(pd.to_numeric)
    df["fuel_type"]=np.where(df["fuel_type"]=="PETROL","petrol",df["fuel_type"])
    df["fuel_type"]=np.where(df["fuel_type"]=="ELECTROCAR","electrocar",df["fuel_type"])
    df["fuel_type"]=np.where(df["fuel_type"]=="DIESEL","diesel",df["fuel_type"])
    df['drive_unit'] = df['drive_unit'].str.lower()
    df['condition'] = df['condition'].str.lower()
    df.drop('car-info', axis='columns', inplace=True)
    df['volume(cm3)'].fillna(df['volume(cm3)'].mean(),inplace=True)
    df['drive_unit'].fillna(df['drive_unit'].mode()[0],inplace=True)
    #df.drop(['color'],axis=1,inplace=True)
    from scipy.stats import normaltest
    y = df['Price Category']
    df.drop(['Price Category'],axis=1,inplace=True)
    
    return df,y

def Feature_Encoder1(X, cols):
    newDataFrame = pd.get_dummies(X,columns=cols)
    #newDataFrame.drop(['fuel_type_electrocar'], axis=1, inplace=True)
    return newDataFrame
     

def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(X[c].values)
        X[c] = lbl.transform(X[c].values)
    return X
     

data = pd.read_csv('cars-train.csv')
cols = ['condition', 'fuel_type', 'transmission', 'drive_unit','mode category','color','segment']
X, y = feature_engineering(data)
col=['brand','model']
Z = Feature_Encoder1(X, cols)
ZZ_X=Feature_Encoder(Z,col)
y.replace({"cheap": 0, "moderate": 1 , "expensive":2 , "very expensive":3}, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(ZZ_X, y, test_size=0.20, random_state=9)
standardScalar = StandardScaler()
standardScalar.fit(X_train[['mileage(kilometers)','volume(cm3)']])
X_train[['mileage(kilometers)','volume(cm3)']] = standardScalar.transform(X_train[['mileage(kilometers)','volume(cm3)']])
X_test[['mileage(kilometers)','volume(cm3)']] = standardScalar.transform(X_test[['mileage(kilometers)','volume(cm3)']])

xgTrain = xgb.DMatrix(X_train,label=y_train)
xgTest = xgb.DMatrix(X_test,label=y_test)

param = {
    'max_depth' :17,#17
    'eta' : 0.07,#0.
    #***************************************
    'colsample_bytree':0.6,#0.6
    'colsample_bylevel':0.6,
    'colsample_bynode':0.6,
    'subsample':1,#1
    'scale_pos_weight':1,
    'refresh_leaf':0,
    'lambda':2,
    #***************************************
    'sampling_method':'uniform',
    'gamma':2,#4
    'min_child_weight':2,#2
    'objective' : 'multi:softmax' ,
    'early_stopping_rounds' : 4,
    'tree_method' : 'exact',
    'num_class' : 4
}
epochs =287#287
xgModel = xgb.train(param,xgTrain,epochs)
xgPredictions = xgModel.predict(xgTest)

print(accuracy_score(y_test,xgPredictions))
print("*********************************************")

print("*********************************************")

     
0.9222135792865874
*********************************************
*********************************************


import pandas as pd
data = pd.read_csv('cars-test.csv')
df = pd.DataFrame(data)


def stringMatchScore (str1 ,str2,str3 , key1 , key2,key3):
    if str1 == key1 and str2==key2:
        return True
    else:
        return False
def stringMatchBrand (str1 , key1):
    if str1 == key1:
        return True
    else:
        return False
def mostLikeStr (str1,str2,str3) :
    flag = 0
    years = []
    for x in brandModelMode:
        newString = x.split(",",3)
        score = stringMatchScore(str1,str2,str3,newString[0],newString[1],newString[2])
        if score :
            years.append(newString[2])
            flag = 1
    min = 100000000000
    if flag :
     for year in range (len(years)):
        word2 = func(str3)
        word1 = func(years[year])
        diff = abs(int(word1)-int(word2))
        if diff < min:
            min = diff
            newstr3 = years[year]
     str3 = newstr3
    if flag:
        return str1+","+str2+","+str3
    return "modelNotFound"
def mostLikeBrand (str1) :
    if brandMode.get(str1,-1) != -1 :
        return brandMode.get(str1)
    return "modelNotFound"

df['volume(cm3)'].fillna(df['volume(cm3)'].mean(), inplace=True)
df['drive_unit'].fillna(df['drive_unit'].mode()[0], inplace=True)
df['segment'].fillna(df['segment'].mode()[0], inplace=True)
li = []
df.drop('car_id', axis='columns', inplace=True)

for n in range(df.shape[0]):
    str = df['car-info'][n]
    strSplit = str.split(",",3)
    if(brandModelMode.get(str, -1) == -1):
        str = mostLikeStr(strSplit[0],strSplit[1],strSplit[2])
        if str == "modelNotFound":
            str = mostLikeBrand(strSplit[1])
            if str != "modelNotFound":
             li.append(str)
             continue
            else:
                li.append(CategoryMode)
                continue
    li.append(brandModelMode.get(str))

df['mode category'] = li
df['mode category'].replace({"cheap": 0, "moderate": 1 , "expensive":2 , "very expensive":3}, inplace=True)
new = df["car-info"].str.split(",", n=3, expand=True)
df['model'] = new[0].apply(func)
df['brand'] = new[1].apply(func)
df['year'] = new[2].apply(func)
df['year'] = df['year'].apply(pd.to_numeric)
df.drop('car-info', axis='columns', inplace=True)
df['fuel_type'] = df['fuel_type'].str.lower()
df['drive_unit'] = df['drive_unit'].str.lower()
df['condition'] = df['condition'].str.lower()
cols = ['condition', 'fuel_type', 'transmission', 'drive_unit','mode category','color','segment']
df1 = Feature_Encoder1(df, cols)
col=['brand','model']
df = Feature_Encoder(df1, col)
standardScalar = StandardScaler()
df[['mileage(kilometers)','volume(cm3)']] = standardScalar.fit_transform(df[['mileage(kilometers)','volume(cm3)']])
subPredictions = xgb.DMatrix(df)
target = xgModel.predict(subPredictions)
target = pd.DataFrame(target)
target.replace({ 0 : "cheap", 1: "moderate" , 2:"expensive" , 3 : "very expensive"}, inplace=True)


ta = pd.read_csv('sample_submission.csv')
ta['Price Category'] = target

ta.to_csv('T5.csv', encoding='utf-8', index=False)

     


     


     


     



     
