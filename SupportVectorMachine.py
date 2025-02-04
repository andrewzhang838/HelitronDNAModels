from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
svm_model = svm.SVC(kernel='linear')  # Linear kernel
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


#SVMs are powerful for classification tasks and can be particularly effective in distinguishing between categories with clear margin separation.
#Differentiating Helitron sequences from other types of transposable elements or non-transposable regions in the genome.