from sklearn import tree
#Weight in grams and bumpy-TRUE OR FALSE
features=[[150,1],[170,1],[140,0],[130,0]]
#Orange-0 and apple -1
labels=[0,0,1,1]


clf=tree.DecisionTreeClassifier()
clf.fit(features,labels)

print(clf.predict([[200,1]]))
