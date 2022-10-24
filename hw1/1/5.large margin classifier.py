import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
# random.seed(123)
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
        train_x = data1.drop(str(train_y.name), axis=1)
        return train_x, train_y

def update_Weight(weight,train_x,train_y,learning_rate=1):

    weight = weight + train_x*train_y*learning_rate

    return weight
def active_function(x):
    if x > 0:
        return 1
    else:
        return -1

def MSE(error,n):
    error = list(map(squre, error))
    return 1/n*sum(error)

def predic(weight,x):
    result = []
    # print(type(x))
    for i in range(x.shape[0]):
        # x = pd.DataFrame(x)
        x1 = np.concatenate((np.array([1.]), np.asarray(x.iloc[i])))
        result.append(active_function(np.dot(weight, x1)))
    return result

def confusion_matrix(result,test_y):
    TP,TN,FT,FN = 0,0,0,0
    # print(result[1])
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



def squre(x):
    return x*x

# def save_wight(x):
#     pass

def euclidean(x,w):

    euc = np.sqrt((np.square(x - w)).sum())

    return euc



def train(train_x,train_y ,epoch = 10):
    w = np.zeros([1, len(train_x.columns) + 1])
    first = True
    n = 0
    while n < epoch:
        error = 0
        n = n + 1
        # mes = []
        save_x = []
        train_y = np.asarray(train_y)
        for i in range(len(train_x)):

            x, y = np.concatenate((np.array([1.]), np.array(train_x.iloc[i]))), np.array(train_y[i])
            save_x.append(x)
            # print(len(save_x))
            func = active_function(np.dot(w, x))
            if y * func <= -1:

                w1 = w
                w2 = update_Weight(w, x, y, 0.01)

                if first:

                    test0 = 0
                    for i in range(len(save_x)):
                        test0 += euclidean(save_x[i],w2)
                    test0 /= len(save_x)
                    w = w2
                    # print(w)
                    first = False
                # print(test0)
                test1 = 0
                for i in range(len(save_x)):
                    test1 += euclidean(save_x[i],w2)
                test1 /= len(save_x)
                # test1 = 1/test1
                # print(test1)
                if test1 < test0:
                    w = w2
                    test0 = test1
                else:
                    w = w1

                error = error + 1

            # mes.append(y-error1)

        # error = MSE(mes,len(train_x))
        if n % 5 == 0:
            print(n, error)
            # print(test0)
            # print(test1)
            # print(train_x)
            # print(w)
    return w

data = pd.read_csv("D:\王艺帆\machine learning\data\data.csv")
lable_class = {
    "M":1,"B":-1
}
data["lable_class"] = data["Diagnosis"].map(lable_class)
data2 = data.drop(["ID","Diagnosis"], axis = 1)
data2 = normalize(data2)

train_x,train_y,test_x,test_y = train_x_train_y(data2,True)


# print(type(test_y))
weight = train(train_x,train_y,30)
y_pred = predic(weight,test_x)
confusion_matrix(y_pred,test_y)

crx =  pd.read_csv("D:\王艺帆\machine learning\data\crx1.csv")
crx = crx.drop(["id"], axis = 1)
crx = crx.dropna()
crx1 = normalize(crx)
# print(crx1.info)
train_x,train_y,test_x,test_y = train_x_train_y(crx1,True)

weight = train(train_x,train_y,30)
y_pred = predic(weight,test_x)
confusion_matrix(y_pred,test_y)
























