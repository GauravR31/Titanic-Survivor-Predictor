import pandas as pd
import matplotlib.pyplot as plt
import numpy
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

pd.set_option('display.max_columns', 20)
passengers_train = pd.read_csv('train.csv')
passengers_test = pd.read_csv('test.csv')

#print(passengers_train.describe(include="all"))

#Male/female survival rate visualization
male_count = ((passengers_train['Sex']=='male').sum())
female_count = ((passengers_train['Sex']=='female').sum())

sexGroup = passengers_train.groupby('Sex')['Survived'].value_counts()

maleSurvivors = sexGroup['male'][1]
femaleSurvivors = sexGroup['female'][1]

plt.bar(['male', 'female'], [(maleSurvivors*100)/male_count, (femaleSurvivors*100)/female_count], width=0.3)
plt.title('Survivors based on sex')
plt.yticks(numpy.arange(0, 100, 5))
#plt.show()

plt.pie([109, 231], labels=['male','female'], autopct='%1.1f%%')
#plt.show()

#Class survival visualization
class1_count = (passengers_train['Pclass']==1).sum()
class2_count = (passengers_train['Pclass']==2).sum()
class3_count = (passengers_train['Pclass']==3).sum()

pClassGroup = passengers_train.groupby('Pclass')['Survived'].value_counts()
class1Survivors = pClassGroup[1][1]
class2Survivors = pClassGroup[2][1]
class3Survivors = pClassGroup[3][1]

plt.bar(['class1', 'class2', 'class3'], 
	[class1Survivors*100/class1_count, class2Survivors*100/class2_count, class3Survivors*100/class3_count])
plt.title('Survivors based on class')
plt.yticks(numpy.arange(0, 100, 5))
#plt.show()

#ParCh survival visualization
parChGroup = passengers_train.groupby('Parch')['Survived'].value_counts()

parCh0_count = (passengers_train['Parch']==0).sum()
parCh1_count = (passengers_train['Parch']==1).sum()
parCh2_count = (passengers_train['Parch']==2).sum()
parCh3_count = (passengers_train['Parch']==3).sum()
parCh4_count = (passengers_train['Parch']==4).sum()
parCh5_count = (passengers_train['Parch']==5).sum()
parCh6_count = (passengers_train['Parch']==6).sum()

parCh0Survivors = parChGroup[0][1]
parCh1Survivors = parChGroup[1][1]
parCh2Survivors = parChGroup[2][1]
parCh3Survivors = parChGroup[3][1]
parCh5Survivors = parChGroup[5][1]
plt.bar(['0', '1', '2', '3', '4', '5', '6'],
	[parCh0Survivors*100/parCh0_count, parCh1Survivors*100/parCh1_count, parCh2Survivors*100/parCh2_count, 
	parCh3Survivors*100/parCh3_count, 0, parCh5Survivors*100/parCh5_count, 0])
plt.yticks(numpy.arange(0, 100, 5))
plt.title('Survivors based on ParentChild')
#plt.show()

#Age survival visualization
sexAge = passengers_train.groupby('Sex')
maleAgeMode = sexAge['Age'].agg(pd.Series.mode)[1].mean()
femaleAgeMode = sexAge['Age'].agg(pd.Series.mode)[0]

sexAgeTest = passengers_test.groupby('Sex')
femaleAgeTestMode = sexAgeTest['Age'].agg(pd.Series.mode)[0]
maleAgeTestMode = sexAgeTest['Age'].agg(pd.Series.mode)[1]
#print(sexAgeTest['Age'].describe())
#print(sexAge['Age'].describe())

passengers_train['Age'] = passengers_train['Age'].fillna(-0.5)
passengers_test['Age'] = passengers_test['Age'].fillna(-0.5)

ageBins = passengers_train.groupby(['Survived', pd.cut(passengers_train['Age'], [-1,0,5,13,18,24,40,60, numpy.inf])])

ageSurvivors = {'Unknown':ageBins.size()[1][0], 'Infant': ageBins.size()[1][5], 'Child':ageBins.size()[1][13], 'Teen':ageBins.size()[1][18], 
'Student':ageBins.size()[1][24], 'Young Adult':ageBins.size()[1][40], 'Adult':ageBins.size()[1][60], 'Senior':ageBins.size()[1][numpy.inf]}

ageDead = {'Unknown':ageBins.size()[0][0],'Infant': ageBins.size()[0][5], 'Child':ageBins.size()[0][13], 'Teen':ageBins.size()[0][18], 
'Student':ageBins.size()[0][24], 'Young Adult':ageBins.size()[0][40], 'Adult':ageBins.size()[0][60], 'Senior':ageBins.size()[0][numpy.inf]}

plt.bar(['Unknown','Infant','Child','Teen','Student','Young Adult','Adult','Senior'],
	[ageSurvivors['Unknown']*100/(ageSurvivors['Unknown']+ageDead['Unknown']),
	ageSurvivors['Infant']*100/(ageSurvivors['Infant']+ageDead['Infant']),
	ageSurvivors['Child']*100/(ageSurvivors['Child']+ageDead['Child']),
	ageSurvivors['Teen']*100/(ageSurvivors['Teen']+ageDead['Teen']),
	ageSurvivors['Student']*100/(ageSurvivors['Student']+ageDead['Student']),
	ageSurvivors['Young Adult']*100/(ageSurvivors['Young Adult']+ageDead['Young Adult']),
	ageSurvivors['Adult']*100/(ageSurvivors['Adult']+ageDead['Adult']),
	ageSurvivors['Senior']*100/(ageSurvivors['Senior']+ageDead['Senior'])],
	color=['red', 'blue', 'green', 'yellow', 'orange', 'black', 'grey'])
plt.yticks(numpy.arange(0, 100, 5))
plt.title('Survivors based on age')
#plt.show()

#Fill missing values for Age
for i in range(0, len(passengers_train['Survived'])):
	if passengers_train['Age'][i] == -0.5:
		if passengers_train['Sex'][i]=='male':
			passengers_train.at[i, 'Age'] = maleAgeMode
		else:
			passengers_train.at[i, 'Age'] = femaleAgeMode

for i in range(0, len(passengers_test['PassengerId'])):
	if passengers_test['Age'][i] == -0.5:
		if passengers_test['Sex'][i]=='male':
			passengers_test.at[i, 'Age'] = maleAgeTestMode
		else:
			passengers_test.at[i, 'Age'] = femaleAgeTestMode
	if math.isnan(passengers_test['Fare'][i]):
		pClass = passengers_test['Pclass'][i]
		passengers_test.at[i, 'Fare'] = passengers_test.groupby('Pclass')['Fare'].mean()[int(pClass)]

passengers_train = passengers_train.drop(['Cabin', 'Ticket'], axis = 1)
passengers_test = passengers_test.drop(['Cabin', 'Ticket'], axis = 1)

passengers_train = passengers_train.fillna({"Embarked": "S"})
passengers_test = passengers_test.fillna({"Embarked": "S"})

embarked_mapping = {"S":1, "C":2, "Q":3}
passengers_train['Embarked'] = passengers_train['Embarked'].map(embarked_mapping)
passengers_test['Embarked'] = passengers_test['Embarked'].map(embarked_mapping)

sex_mapping = {"male":0, "female":1}
passengers_train['Sex'] = passengers_train['Sex'].map(sex_mapping)
passengers_test['Sex'] = passengers_test['Sex'].map(sex_mapping)

survived_train = passengers_train['Survived']
passengers_train = passengers_train.drop(['Survived', 'PassengerId', 'Name'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(passengers_train, survived_train, test_size = 0.25, random_state = 42)
#Regression, SVC, KNearest, NaiveBayes, DecisionTree, RandomForest
'''logRegression = LogisticRegression()
logRegression.fit(X_train, Y_train)

logRegressionPred = logRegression.predict(X_test)
print("Logistic Regression", accuracy_score(Y_test, logRegressionPred), precision_score(Y_test, logRegressionPred),
	recall_score(Y_test, logRegressionPred))

svc = SVC(gamma = 'scale', C=100, kernel = 'linear')
svc.fit(X_train, Y_train)

svcPred = svc.predict(X_test)
print("SVC", accuracy_score(Y_test, svcPred), precision_score(Y_test, svcPred),
	recall_score(Y_test, svcPred))

knn = KNeighborsClassifier(n_neighbors=18)
knn.fit(X_train, Y_train)

knnPred = knn.predict(X_test)
print("KNN", accuracy_score(Y_test, knnPred), precision_score(Y_test, knnPred),
	recall_score(Y_test, knnPred))

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

gnbPred = gnb.predict(X_test)
print("Gaussian NB", accuracy_score(Y_test, gnbPred), precision_score(Y_test, gnbPred),
	recall_score(Y_test, gnbPred))'''

dtClassifier = DecisionTreeClassifier(min_samples_split=16, max_depth = 16)
dtClassifier.fit(X_train, Y_train)

dtClassifierPred = dtClassifier.predict(X_test)
print("Decision Tree", accuracy_score(Y_test, dtClassifierPred), precision_score(Y_test, dtClassifierPred),
	recall_score(Y_test, dtClassifierPred))

'''randomForest = RandomForestClassifier(n_estimators=20, max_depth=4)
randomForest.fit(X_train, Y_train)

randomForestPred = randomForest.predict(X_test)
print("Random Forest", accuracy_score(Y_test, randomForestPred), precision_score(Y_test, randomForestPred),
	recall_score(Y_test, randomForestPred))'''

ids = passengers_test['PassengerId']
predictions = dtClassifier.predict(passengers_test.drop(['PassengerId', 'Name'], axis = 1))

output = pd.DataFrame({'PassengerId' : ids, 'Survived' : predictions})
output.to_csv('submission1.csv', index = False)