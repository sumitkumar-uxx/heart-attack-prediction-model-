import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , recall_score , f1_score , accuracy_score
a = pd.read_csv(r"C:\Users\sumit\OneDrive\Documents\heart.csv")
b = pd.DataFrame(a)
c = b.copy()
k = LabelEncoder()
c["sex"]=k.fit_transform(c["sex"])
c["diabetes"]=k.fit_transform(c["diabetes"])
ck = c[["age","cholesterol","bp","sex","diabetes"]]
d = c["heart_disease"]
e = LogisticRegression()
x_train , x_test ,y_train , y_test = train_test_split(ck,d , train_size=0.8 , test_size= 0.2 , random_state=42)
e.fit(x_train , y_train)
gotta = e.predict(x_test)
print("precsiosn score:", precision_score(gotta,y_test , average='macro'))
print("acuuracy score:", accuracy_score(gotta, y_test))
print("recall score:" ,  recall_score(gotta, y_test , average='macro'))
print("f1 score: " ,  f1_score(gotta, y_test,average='macro'))
print("heart testing started.........")
f = int(input("enter you age:"))
g = int(input("enter your cholesterol:"))
h = int(input("enter your bp:"))
i = int(input("enter your sex:"))
j = int(input("enter your diabetes:"))
jk = e.predict([[f,g,h,i,j]])
if jk == 1:
    print("heart acttack detected.")
else:
    print("heart attack not detected.")
