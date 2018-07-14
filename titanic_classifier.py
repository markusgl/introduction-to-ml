""" TITANIC kaggle competition """

import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib

survived = {0: 'drown', 1: 'survive'}
gender = {'male': 1,'female': 2}
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# prepare training data
df = pd.read_csv('data/titanic/train.csv')
df.dropna(how='any', inplace=True)

data = df[features]
labels = df['Survived']

X_train = []
for i, row in data.iterrows():
    if row[1] == 'male':
        row[1] = gender['male']
    else:
        row[1] = gender['female']

    X_train.append(row)
    #print(row)

# Train the classifier
clf = SVC(kernel='linear')
print("start training...")
clf.fit(X_train, labels)

# safe model with pickle
joblib.dump(clf, 'titanic_classifier.pkl')
# load classifier
#clf = joblib.load('titanic_classifier.pkl')


###### VALIDATION #######
# prepare test data
df_test = pd.read_csv('data/titanic/test.csv')
df_test.dropna(how='any', inplace=True)
test_data = df_test[features]

test_labels = pd.read_csv('data/titanic/gender_submission.csv')

X_test = []
for i, row in test_data.iterrows():
    if row[1] == 'male':
        row[1] = 1
    else:
        row[1] = 2

    X_test.append(row)

y_test = []
for i, row in test_labels.iterrows():
    if df_test['PassengerId'].isin([row[0]]).any():
        y_test.append(row['Survived'])

score = clf.score(X_test, y_test)
print("Score: %.3f" % score)


# test with examples
passenger1 = {'Pclass': 1, 'Sex': gender['male'], 'Age': 95, 'SibSp': 0, 'Parch': 0}
passenger2 = {'Pclass': 2, 'Sex': gender['female'], 'Age': 44, 'SibSp': 0, 'Parch': 0}
passenger3 = {'Pclass': 3, 'Sex': gender['female'], 'Age': 14, 'SibSp': 0, 'Parch': 0}

passenger_list = []
passenger_list.append(pd.Series(passenger1))
passenger_list.append(pd.Series(passenger2))
passenger_list.append(pd.Series(passenger3))
predictions = clf.predict(passenger_list)

for i in range(len(predictions)):
    print("Passenger {} will {}!".format(i, survived[predictions[i]]))


# TODO hyperparameter estimation
