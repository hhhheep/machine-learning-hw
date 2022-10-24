from sklearn.svm import SVC
import pandas as pd
import numpy as np

def normalize(data):

    nm = data.mean()
    # print(nm)
    # print(data.shape)
    for i in range(data.shape[1]-1):
        data.iloc[:,i] = data.iloc[:,i]/nm[i]
    # print(data)
    return data

def train_x_train_y(data,wanna_test = False,test_num = 0.25):

    data1 = data.iloc[np.random.permutation(len(data))]
    if wanna_test :

        data2 = data1[0:round((1-test_num)*len(data))]
        train_y = data2.iloc[:,-1]
        train_x = data2.drop(str(train_y.name), axis = 1)
        data3 = data1[round((1-test_num)*len(data)):]
        test_y = data3.iloc[:,-1]
        test_x = data3.drop(str(test_y.name), axis = 1)

        return train_x,train_y,test_x,test_y

    else:
        train_y = data1.iloc[:, -1]
        train_x = data1.drop(str(train_y.name), axis = 1)
        return train_x, train_y

def confusion_matrix(result,test_y):
    TP,TN,FT,FN = 0,0,0,0
    # print(result[1])
    result = np.asarray(result)
    test_y = np.asarray(test_y)
    # print(test_y[1])
    for i in range(len(result)):
        if result[i] > 0 :
            if result[i] * test_y[i] >=1:
                TP += 1
            else:
                TN += 1
        else:
            if result[i] * test_y[i] >=1:
                FT += 1
            else:
                FN += 1
    accuracy = (TP + FT) / len(result)
    cm = np.array([[TP,FN],[TN,FT]])
    print("confusion_matrix:")
    print(cm)
    print("accuracy:",accuracy)


data = pd.read_csv("D:\王艺帆\machine learning\data\data.csv")
lable_class = {
    "M":1,"B":-1
}
data["lable_class"] = data["Diagnosis"].map(lable_class)
data2 = data.drop(["ID","Diagnosis"], axis = 1)
data2 = normalize(data2)

train_x,train_y,test_x,test_y = train_x_train_y(data2,True)

svm = SVC()
svm.fit(train_x,train_y)
svm.score(test_x,test_y)
confusion_matrix(svm.predict(test_x),test_y)

crx =  pd.read_csv("D:\王艺帆\machine learning\data\crx1.csv")
crx = crx.drop(["id"], axis = 1)
crx = crx.dropna()
crx1 = normalize(crx)
# print(crx1.info)
train_x,train_y,test_x,test_y = train_x_train_y(crx1,True)
# print(train_y);print(train_x)
svm = SVC()
svm.fit(train_x,train_y)
# svm.score(test_x,test_y)
confusion_matrix(svm.predict(test_x),test_y)
