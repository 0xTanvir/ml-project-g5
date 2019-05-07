# scikit-learn :
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


Classifiers = [
    SVC(kernel='rbf', probability=True, C=5000),
    RandomForestClassifier(n_estimators=300)
]


def runClassifiers(X, y):
    Results = []

    from sklearn.metrics import accuracy_score

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    
    importance_forest = RandomForestClassifier()

    for classifier in Classifiers:

        accuray = []
     

        print(classifier.__class__.__name__)

        model = classifier
        for (train_index, test_index) in cv.split(X, y):

            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]
            
            model.fit(X_train, y_train)

            y_artificial = model.predict(X_test)

           
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            


        accuray = [_*100.0 for _ in accuray]
        Results.append(accuray)

        
        print('Accuracy: {0:.4f} %'.format(np.mean(accuray)))


        print('_______________________________________')

#lets load the dataset 
X_AAC = np.load("X_train_AAC.npy")
X_CKSAAP = np.load("X_train_CKSAAP.npy")
X_GAAC = np.load("X_train_GAAC.npy")
X_CKSAAGP = np.load("X_train_CKSAAGP.npy") 

X = np.concatenate((X_CKSAAGP,X_AAC,X_GAAC),axis=1)
Y = np.load("train_labels.npy")

runClassifiers(X, Y)


