from sklearn.naive_bayes import MultinomialNB

from loader import load_dataset, to_col, gen_folders
from sklearn.calibration import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import json

train_set, test_set = load_dataset()


x, y = to_col(train_set)

for (i, label) in zip(x[:1000], y[:1000]):
    print(f"label: {label}, text: {i}")

print(f"train size: {len(x)}")
print("training TF-IDF...")
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
X_train = tfidf.fit_transform(x)

print("training SVM...")
svc = LinearSVC()
# svc = MultinomialNB()
svc.fit(X_train, y)

x_test, y_test = to_col(test_set)
print(f"test size: {len(x_test)}")
print("evaluating...")
X_test = tfidf.transform(x_test)
y_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
print(
    classification_report(
        y_test, y_pred, labels=list(range(0, len(gen_folders) + 1)), target_names=["human"] + list(gen_folders.keys()), digits=4
    )
)

