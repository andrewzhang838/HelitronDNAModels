from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X and y are your features and labels respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


#This model is good for classification and regression tasks and can handle large datasets with high dimensionality, making it suitable for genomic data.
#Classifying genomic regions as Helitrons or non-Helitrons based on a set of features derived from the genomic sequences, such as nucleotide composition, presence of certain motifs, or similarity to known Helitrons.
