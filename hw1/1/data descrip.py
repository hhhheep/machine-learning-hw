import pandas as pd
import numpy as np
data = pd.read_csv("D:\王艺帆\machine learning\data\crx.csv")

# print(np.where(data == "?")[0])
data = data.drop(np.where(data == "?")[0])
# for cat_fea in data:
#     try:
#         print(cat_fea + "的特徵分佈如下：")
#         print("{}特徵有个{}不同的值".format(cat_fea, data[cat_fea].nunique()))
#         print(data[cat_fea].value_counts())
#     except:
#         continue
lable_att1 = {
    "a":1,"b":0
}
lable_att4 = {
    "u":1,"y":0,"k":-1
}
lable_att5 = {
    "g":1,"p":0,"gg":-1
}
lable_att6 = {
    "c":1,"q":2,"w":3,"i":4,"aa":5,"ff":6,"k":7,"cc":8,"m":9,"x":10,"d":11,"e":12,"j":13,"r":14
}
lable_att7 = {
    "v":1,"h":2,"ff":3,"bb":4,"j":5,"z":6,"dd":7,"n":8,"o":9
}

lable_att9 = {
    "t":1,"f":0
}
lable_att10 = {
    "t":1,"f":0
}
lable_att12 = {
    "t":1,"f":0
}
lable_att13 = {
    "g":1,"s":-1,"p":0
}

lable_class = {
    "+":1,"-":-1
}
data["att1"] = data["att1"].map(lable_att1)
data["att4"] = data["att4"].map(lable_att4)
data["att5"] = data["att5"].map(lable_att5)
data["att6"] = data["att6"].map(lable_att6)
data["att7"] = data["att7"].map(lable_att7)
data["att9"] = data["att9"].map(lable_att9)
data["att10"] = data["att10"].map(lable_att10)
data["att12"] = data["att12"].map(lable_att12)
data["att13"] = data["att13"].map(lable_att13)
data["lable_class"] = data["label"].map(lable_class)
data = data.drop(["label"], axis = 1)
data.to_csv("D:\王艺帆\machine learning\data\crx1.csv")

for cat_fea in data:
    try:
        print(cat_fea + "的特徵分佈如下：")
        print("{}特徵有个{}不同的值".format(cat_fea, data[cat_fea].nunique()))
        print(data[cat_fea].value_counts())
    except:
        continue