import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv("C:\\Users\\dell\\Desktop\\ML\\Xy_train.csv")
df = df.astype(dtype= {"age":"float", "gender":"category","cp":"category", "trestbps":"float", "chol":"float","fbs":"category", "restecg":"category", "thalach":"int64","exang":"category", "oldpeak":"float","slope":"category","ca":"category","thal":"category","y":"category"})
# -------------------Age------------------- #
df.query('age >= 100').age.count()
AgeMean0 = df[(df['y'] == 0) & (df['age'] < 100)]['age'].mean().round()
AgeMean1 = df[(df['y'] == 1) & (df['age'] < 100)]['age'].mean().round()
df.loc[(df['y'] == 1) & (df['age'] > 100), 'age'] = AgeMean1
df.loc[(df['y'] == 0) & (df['age'] > 100), 'age'] = AgeMean0

#------------------- CA-------------------- #
type(df['thal'])
df.loc[(df['ca'] == 4),'ca'] = 0

#------------------- THAL-------------------- #
df.loc[(df['thal'] == 0),'thal'] = 2

# -------------Create age categories---------------- #
labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]
df['AgeGroup'] = pd.cut(df['age'], range(0, 105, 10), right=False, labels=labels)
df['AgeGroup'] = df['AgeGroup'].astype('category')

#---------------------Pre-processing-------------------#
X_train = df.drop(['y','age','id'], 1) # data for DT
Y_train = df['y'] # labels
X_train = pd.get_dummies(X_train)
#--------------------Feature Normalization---------------#
scaler = StandardScaler()
numeric_cols = X_train[['trestbps', 'chol', 'thalach', 'oldpeak']]
categorial_cols = X_train.drop(['trestbps', 'chol', 'thalach', 'oldpeak'],1)
normalized_data = scaler.fit_transform(numeric_cols)
normalized_data = pd.DataFrame(normalized_data, columns = numeric_cols.columns)
X_train_Norm = normalized_data.join(categorial_cols) # data with normalized numeric vales for ANN
X_train_Norm = pd.get_dummies(X_train_Norm)
#---------------feature selection--------------#
#normalized_data = X_train_Norm.drop('fbs',1)
#data = X_train.drop('fbs',1)

#----------------dimensionality reducation (PCA)---------#


#-----------------DT----------------------------#
from sklearn.model_selection import cross_val_score
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, Y_train)
#plt.figure(figsize=(8,7))
#plot_tree(model, filled = True, class_names = True)
#plt.show()
scores = cross_val_score(model, X_train, Y_train, cv=10)
print(" Validation Accuracy: %0.2f " % (scores.mean()))
print("Train accuracy: ", round(model.score(X_train, Y_train), 2))

#--------------Hyperparameters Tuning-------------------#

param_grid = {'max_depth': np.arange(1, 12, 1),
              'criterion': ['entropy', 'gini'],
              'ccp_alpha': np.arange(0, 1, 0.05)
             }
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, refit=True, cv=10, return_train_score = True)# need to add train score
grid_search.fit(X_train, Y_train)
Results =pd.DataFrame(grid_search.cv_results_)
y = Results['mean_test_score']
plt.plot(y)
plt.xlabel('Iteration')
plt.ylabel('Validation accuracy')
plt.title('Accuracy for each Iteration')
plt.show()
table = pd.DataFrame(Results).sort_values('rank_test_score')[['params','mean_test_score']]