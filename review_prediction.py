import pandas as pd
df = pd.read_csv('moviereviews2.tsv',sep = '\t')

blanks = []
for ind,label,review in df.itertuples():
    if type(review) == str:
        if review.isspace():
            blanks.append(ind)
df.dropna(inplace = True)
df['label'].value_counts()
from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
text_classifier = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
text_classifier.fit(X_train,y_train)
predictions = text_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
