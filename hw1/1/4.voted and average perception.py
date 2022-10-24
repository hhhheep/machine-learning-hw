import numpy as np
import pandas as pd
import operator
class average_vote_perceptron(object):
    def update_Weight(self,weight,train_x,train_y,learning_rate=1):

        weight = weight + train_x*train_y*learning_rate

        return weight
    def active_function(self,x):
        if x > 0:
            return 1
        else:
            return -1

    def MSE(self,error,n):
        error = list(map(self.squre, error))
        return 1/n*sum(error)

    def predict(self,x,type = None):
        result = []

        x = pd.DataFrame(x)
        average1 = self.average()
        # print(average1)
        # x = np.asarray(x)
        # print(x[1][1])
        for i in range(x.shape[0]):
            x1 = np.concatenate((np.array([1.]), np.array(x.iloc[i])))
            if type == "vote":
                result.append(self.vote(x1))
                # print(result)

            if type == "average":
                result.append(self.active_function(np.dot(average1, x1)))
                # print(result)

        return result

    def squre(self,x):
        return x*x

    def get_wight(self):
        return (self.weight,self.c)

    def vote(self,x):
        vote_dict = {}
        n = 0
        for i in self.save_weight:

            result = self.active_function(np.dot(i,x))
            # print(self.c)
            if result not in vote_dict.keys():
                vote_dict[result] = self.c[n]
            else:
                # print(vote_dict)
                vote_dict[result] += self.c[n]
            n += 1
        sort_vote_disc = sorted(vote_dict.items(),key=operator.itemgetter(1),reverse=True)

        return sort_vote_disc[0][0]

    def average(self):

        average_weight = np.sum(self.save_weight, axis = 0)/sum(self.c)
        return average_weight

    def train(self,train_x,train_y,epcho=10):

        self.weight = np.zeros([1, train_x.shape[1]+ 1])
        error = 0
        self.save_weight = []
        # mes = []
        self.c = []
        k = 0
        train_y = np.asarray(train_y)
        n = 0
        while n < epcho:
            n += 1
            for i in range(len(train_x)):
                x, y = np.concatenate((np.array([1.]), np.array(train_x.iloc[i]))), np.array(train_y[i])
                func = self.active_function(np.dot(self.weight, x))
                k = k+1
                if func * y <= -1:
                    self.weight = self.update_Weight(self.weight, x, y, 0.2)
                    self.save_weight.append(self.weight)
                    self.c.append(k)
                    k = 0
                    error = error + 1

def normalize(data):
    nm = data.mean()
    for i in range(data.shape[1] - 1):
        data.iloc[:, i] = data.iloc[:, i] / nm[i]
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

avp = average_vote_perceptron()
avp.train(train_x,train_y)
# avp.predict(test_x,"vote")
confusion_matrix(avp.predict(test_x,"vote"),test_y)
confusion_matrix(avp.predict(test_x,"average"),test_y)

crx =  pd.read_csv("D:\王艺帆\machine learning\data\crx1.csv")
crx = crx.drop(["id"], axis = 1)
crx = crx.dropna()
crx1 = normalize(crx)
# print(crx1.info)
train_x,train_y,test_x,test_y = train_x_train_y(crx1,True)
# train_x,train_y,test_x,test_y = np.asarray(train_x),np.asarray(train_y),np.asarray(test_x),np.asarray(test_y)

avp = average_vote_perceptron()
avp.train(train_x,train_y)
# avp.predict(test_x,"vote")
confusion_matrix(avp.predict(test_x,"vote"),test_y)
confusion_matrix(avp.predict(test_x,"average"),test_y)