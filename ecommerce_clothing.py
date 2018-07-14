''' E-Commerce Clothing'''
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sentiment = {0: 'negative', 1: 'positive'}

df = pd.read_csv("data/review_data.csv", usecols=['Review Text', 'Rating'])
print(df.head())
df.dropna(how="any", inplace=True)
print("rows %i" % len(df))

#reviews = df[['Review Text', 'Rating']]
#reviews.dropna(how="any", inplace=True)
text = []
labels = []
positive =[]
negative = []
#for i, row in reviews.iterrows():
for i, row in df.iterrows():
    text.append(row['Review Text'])
    if row['Rating'] >= 3:
        labels.append(1) # positive review
        positive.append(1)
    else:
        labels.append(0) # negative review
        negative.append(1)

# Convert the text into vectors
#vectorizer = CountVectorizer()
print(len(positive))
print(len(negative))
vectorizer = TfidfVectorizer(stop_words='english')
counts = vectorizer.fit_transform(text)
print("vocabulary length %i" % len(vectorizer.vocabulary_.keys()))
vocabulary = vectorizer.vocabulary_
#print(counts)
#print(vocabulary)


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(counts, labels, test_size=0.2, random_state=1)
"""
X = vectorizer.fit_transform(X_train).todense()
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:, 0], data2D[:, 1], c=y_train)
plt.show()
"""

# Train the SVM
clf = SVC(kernel='rbf', C=100, gamma=0.01, decision_function_shape='ovo', probability=True)
print("start training...")
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

# safe model with pickle
joblib.dump(clf, 'eclothing_classifier.pkl')
