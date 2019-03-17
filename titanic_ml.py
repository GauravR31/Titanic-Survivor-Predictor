import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

passengers_train = pd.read_csv('train.csv')
passengers_test = pd.read_csv('test.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

for i in range(0, len(passengers_train['Survived'])):
	if math.isnan(passengers_train['Age'][i]):
		passengers_train.at[i, 'Age'] = passengers_train['Age'].mean()

	if passengers_train['Sex'][i] == 'male':
		passengers_train.at[i, 'Sex'] = 1

	if passengers_train['Sex'][i] == 'female':
		passengers_train.at[i, 'Sex'] = 0
				
	if math.isnan(passengers_train['Fare'][i]):
		passengers_train.at[i, 'Fare'] = passengers_train['Fare'].mean()

for i in range(0, len(passengers_test['Sex'])):
	if math.isnan(passengers_test['Age'][i]):
		passengers_test.at[i, 'Age'] = passengers_test['Age'].mean()

	if passengers_test['Sex'][i] == 'male':
		passengers_test.at[i, 'Sex'] = 1

	if passengers_test['Sex'][i] == 'female':
		passengers_test.at[i, 'Sex'] = 0

	if math.isnan(passengers_test['Fare'][i]):
		passengers_test.at[i, 'Fare'] = passengers_test['Fare'].mean()	

X = passengers_train[features]
Y = passengers_train['Survived']

selector = SelectKBest(chi2, k = 4)
selector.fit_transform(X, Y)
print selector.get_support()

my_features = ['Pclass', 'Sex', 'Age', 'Fare']

'''for i in range(0, len(passengers_train['Survived'])):
	x = passengers_train['Age'][i]
	y = passengers_train['Fare'][i]

	plt.scatter(x, y)

plt.show()'''

X = passengers_train[my_features]

X_test = passengers_test[my_features]
X_id = passengers_test['PassengerId']

for i in range(0, len(X_test['Sex'])):
	if math.isnan(X_test['Sex'][i]):
		print 'Sex', i

	if math.isnan(X_test['Age'][i]):
		print 'Age', i

	if math.isnan(X_test['Fare'][i]):
		print 'Fare', i

features_train, features_test, labels_train, labels_test = train_test_split(
	X, Y, test_size = 0.3, random_state = 43)

dt_clf = DecisionTreeClassifier(min_samples_split=9, criterion='gini')
dt_clf.fit(features_train, labels_train)

dt_pred = dt_clf.predict(features_test)
print "DecisionTreeClassifier ", accuracy_score(labels_test, dt_pred), precision_score(labels_test, dt_pred), recall_score(labels_test, dt_pred)

dt_test_pred = dt_clf.predict(X_test)
dt_test_pred = pd.DataFrame(dt_test_pred)

X_id.to_csv('results.csv', index = None, header = 'PassengerId')
df_csv = pd.read_csv('results.csv')
df_csv['Survived'] = dt_test_pred
df_csv.to_csv('results.csv', index = None)

'''svc = SVC()
svc.fit(features_train, labels_train)

svc_pred = svc.predict(features_test)

print "SVC ", accuracy_score(labels_test, svc_pred), precision_score(labels_test, svc_pred), recall_score(labels_test, svc_pred)

naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(features_train, labels_train)

naive_bayes_pred = naive_bayes_clf.predict(features_test)

print "GaussianNB ", accuracy_score(labels_test, naive_bayes_pred), precision_score(labels_test, naive_bayes_pred), recall_score(labels_test, naive_bayes_pred)'''