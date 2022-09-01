import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA


df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/datas/breast-cancer-data.csv")
df.head()

"""1.   Diagnosis: Tanı
2.   Radius: Yarıçap
3.   Texture: Doku
4.   Perimeter: Çevre
5.   Area: Alan
6.   Smoothness: Düzgünlük
7.   Compactness: Kompaktlık
8.   Concavity: İçBükeylik
9.   Concave Points: İçBükey Noktalar
10.  Symmetry: Simetri
11.  Fractal Dimension: Fraktal Boyut

---
Her görüntü için bu özelliklerin ortalaması, standart hatası ve “en kötü” 
veya en büyüğü (en büyük üç değerin ortalaması) hesaplandı ve 30 özellik elde edildi. Örneğin, alan 3, Ortalama Yarıçap, alan 13, Yarıçap SE, alan 23, En Kötü Yarıçaptır.
"""

df.info()

df = df.drop("id",axis=1)

df.describe().T

sns.pairplot(df)

plt.figure(figsize=(25,10))
cmap = sns.diverging_palette(260, 10, as_cmap=True)
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot = True, fmt = ".2f", cmap = cmap, mask = mask)
plt.show()

sns.countplot(df.diagnosis)
plt.title("Diagnosis")
df.diagnosis.value_counts()


def colSplit(df,limit):
  num_cols = [col for col in df.columns if df[col].dtype != "O" and df[col].nunique() > limit]
  cat_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() <= limit]
  cat_but_num = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() > limit]
  num_but_cat = [col for col in df.columns if df[col].dtype != "O" and df[col].nunique() <= limit]
  cat_cols = cat_cols + num_but_cat
  return cat_cols, num_cols, cat_but_num
cat_cols, num_cols, cat_but_num = colSplit(df,10)
print("Categoric Columns: {}\nNumeric Columns: {}\nCategoric but Numeric Columns: {}".format(cat_cols, num_cols, cat_but_num))

fig, axes = plt.subplots(nrows = 1, ncols = 3, sharex = True, figsize = (20,6))
sns.scatterplot(df.diagnosis, df.fractal_dimension_worst, ax = axes[0])
sns.scatterplot(df.diagnosis, df.fractal_dimension_se, ax = axes[1])
sns.scatterplot(df.diagnosis, df.fractal_dimension_mean, ax = axes[2])

df.columns

for col in num_cols:
    plt.tight_layout()
    fig, (ax_box, ax_hist) = plt.subplots(1,2, sharex = True, figsize = (15,5))
    sns.boxplot(df[col], ax = ax_box, linewidth = 1)
    sns.distplot(df[col], ax = ax_hist)

"""for col in num_cols: 
    quantile = [0.1,0.25,0.35,0.4,0.55,0.63,0.72,0.85,0.95,0.99]
    print(df[col].describe(quantile))
"""

def outlierThreshold(df,col,q1,q3):
    Q1 = df[col].quantile(q1)
    Q3 = df[col].quantile(q3)
    iqr = Q3 - Q1
    lowerLimit = Q1 - iqr * 1.5
    upperLimit = Q3 + iqr * 1.5
    return lowerLimit, upperLimit
def isThereOutlier(df,col, q1, q3):
    lowerLimit, upperLimit = outlierThreshold(df,col, q1, q3)
    outlierDf = df.loc[(df[col] < lowerLimit) | (df[col] > upperLimit)]
    if outlierDf.any(axis = None):
        return True
    else: 
        return False 
    
outlierCols = []
for col in num_cols:
    print(col,": ", isThereOutlier(df, col, q1 = 0.25, q3 = 0.75))
    if isThereOutlier(df, col, q1 = 0.25, q3 = 0.75) == True:
        outlierCols.append(col)

for col in outlierCols:
    lowerLimit, upperLimit = outlierThreshold(df, col, q1 = 0.25, q3 = 0.75)
    #print("lowerLimit: {}\nupperLimit: {}".format(lowerLimit,upperLimit))
    df = df.loc[(df[col] > lowerLimit) & (df[col] < upperLimit)]
for col in outlierCols:
    print(col,": ", isThereOutlier(df, col, q1 = 0.25, q3 = 0.75))

def catchOutliers(df,col, plot=False,q1=0.25, q3=0.75):
    if plot:
       sns.boxplot(df[col])
       plt.title(col)
       plt.show()
    lowerLimit, upperLimit = outlierThreshold(df, col, q1 = 0.25, q3 = 0.75)
    valuesDf = df[(df[col] < lowerLimit) | (df[col] > upperLimit)]
    if valuesDf.shape[0] > 10:
       return str(col) + "\n" + str(valuesDf.head()) + "\n\n" + str(valuesDf.index)
    else:
       return str(col) + "\n" + str(valuesDf) + "\n\n" + str(valuesDf.index)
for col in outlierCols:
    print(catchOutliers(df,col,plot = True, q1 = 0.25, q3 = 0.75))

df.head()

labelEncoder = LabelEncoder()
df["diagnosis"] = labelEncoder.fit_transform(df["diagnosis"])
df.head()

def trainTest(df,target):
    X = df.drop(target, axis = 1).values
    y = df[target].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def models():
  classifier = [
      ("LogisticRegression",LogisticRegression()),
      ("GaussianNB",GaussianNB()),
      ("DecisionTreeClassifier",DecisionTreeClassifier()),
      ("MLPClassifier",MLPClassifier()),
      ("KNeighborsClassifier",KNeighborsClassifier()),
      ("GradientBoostingClassifier",GradientBoostingClassifier()),
      ("LGBMClassifier",LGBMClassifier()),
      ("AdaBoostClassifier",AdaBoostClassifier())
  ]
  return classifier

def modelStep(cls, X_train, X_test, y_train, y_test, y_pred, name): # model asamalari
       cv_results = cross_val_score(cls,X_train, y_train, cv = 10, scoring = "accuracy")
       cv_mean = cv_results.mean()
       mse = mean_squared_error(y_test, y_pred)
       mae = mean_absolute_error(y_test,y_pred)
       rmse = np.sqrt(mean_squared_error(y_test, y_pred))
       r2_Score = r2_score(y_test, y_pred)
       #realY = pd.DataFrame(y_test, columns = ["Real_Y"])
       #predictY = pd.DataFrame(y_pred, columns = ["Predict_Y"])
       #resultDf = pd.concat([realY, predictY], axis = 1)
       print("Model: {}\nMSE: {}\nMAE: {}\nCV Mean: {}\nR2 Score: {}\nRMSE: {}\n".format(name, mse, mae, cv_mean, r2_Score, rmse))

def modelRun(df, target, pcaA = False, pca_number = 2):
    X_train, X_test, y_train, y_test = trainTest(df,target)
    classifier = models()
    for name, model in classifier:
       cls = model.fit(X_train, y_train)
       y_pred = cls.predict(X_test)
       print("**************************** PCA OLMADAN {} ************************".format(name))
       print("X_train shape: {}\nX_test shape: {}\n".format(X_train.shape, X_test.shape))
       modelStep(cls,X_train, X_test, y_train, y_test, y_pred, name)  

       if pcaA == True:
            print("**************************** PCA {} ************************".format(name))
            pca = PCA(n_components = pca_number)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            cls = model.fit(X_train_pca, y_train)
            y_pred_pca = cls.predict(X_test_pca)  
            print("X_train_pca shape: {}\nX_test_pca shape: {}\n".format(X_train_pca.shape, X_test_pca.shape))
            modelStep(cls,X_train_pca, X_test_pca, y_train, y_test, y_pred_pca, name)

    #return resultDf
modelRun(df,"diagnosis", pcaA = True, pca_number = 2)

"""Logistic Regression ve KNeighborsClassifier iyi bir sonuc vermistir."""

from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = trainTest(df,"diagnosis")
knn = KNeighborsClassifier()
grid_params = {
    "n_neighbors":[3,5,10,15],
    "weights":["uniform","distance"],
    "algorithm":["auto","ball_tree","kd_tree","brute"],
    "metric":["euclidean","manhattan"]

}
gscv = GridSearchCV(estimator = knn, param_grid = grid_params, scoring = "accuracy", cv = 10)
gscv.fit(X_train, y_train)
bestParams = gscv.best_params_
bestResult = gscv.best_score_
print("Best Params: {}\nBest Result: {}".format(bestParams, bestResult))

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=3, algorithm="auto", metric = "euclidean", weights="uniform")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_Score = r2_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = ".2f")
plt.title("KNeighborsClassifier")
plt.show()
print("\nMSE: {}\nMAE: {}\nRMSE: {}\nR2 Score: {}\nF1 Score: {}\nAccuracy Score: {}\n".format(mse, mae, rmse, r2_Score,f1_score, acc_score))

