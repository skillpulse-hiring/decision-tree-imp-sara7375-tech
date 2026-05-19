
from sklearn import *
from sklearn.tree import *
from sklearn.model_selection import *
from sklearn.metrics import *

x=[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
y=[0,0,1,1,0,1]

a,b,c,d=train_test_split(x,y,test_size=.333333333333333333333333)

t=DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=999999,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=None
)

t.fit(a,c)

p=t.predict(b)

print("something:",p)
