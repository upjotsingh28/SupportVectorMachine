import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Loading Sample Data
X, y = make_blobs(n_samples=100, centers=2, random_state=4)

def SVM_Example():
    print("SVM HyperPlane Classifier Example with C(Regularisation Parameter) = 10 :")
    print("Plotting the data points and SVM generated hyperplane...")
    Model = SVC(kernel='linear', C=10)
    Model.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.coolwarm)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = Model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(Model.support_vectors_[:, 0], Model.support_vectors_[:, 1], s=300,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

SVM_Example()

print("Apply this model on Breast Cancer DataSet with different C and Gamma Values:\nWe Have our DataSet as :\n")

cancerData = load_breast_cancer()

# Getting the features of data
df_feat = pd.DataFrame(cancerData['data'], columns=cancerData['feature_names'])

print(df_feat.head())

Target = cancerData['target']

XFeature = df_feat
yTarget = Target

X_test, x_train, Y_test, y_train = train_test_split(XFeature, yTarget, test_size=0.33, random_state=8)

print("Applying GridSearch for getting best params for SVM Model.\n")
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3,cv=3)
grid.fit(XFeature, yTarget)

print("\nWe got our best parameters as : {}".format(grid.best_params_))
print("Applying this parameters onto our Model, We got Confusion Matrix :\n")
gridPredictions = grid.predict(X_test)
print(confusion_matrix(Y_test, gridPredictions),"\n\nClassification Report :\n")
print("\n", classification_report(Y_test, gridPredictions))
