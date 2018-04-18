# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
from sklearn.learning_curve import learning_curve
from sklearn import linear_model
from sklearn import cross_validation
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"trainning set")
        plt.ylabel(u"score")
        plt.gca()
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"training set score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cross validation score")

        plt.legend(loc="best")

#        plt.draw()
#        plt.show()
#        plt.gca()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def set_missing_age_byTitle(df,map_means):
           
   idx_nan_age = df.loc[np.isnan(df['Age'])].index
   df['Age'].loc[idx_nan_age] = df['Title'].loc[idx_nan_age].map(map_means)
   return df
  

def process_data(data_train,map_means):
    data_train = set_Cabin_type(data_train)
        
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
    
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
    
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
    
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
    
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df.drop('SibSp',axis=1,inplace=True)
    df.drop('Parch',axis=1,inplace=True)
    
    df=set_missing_age_byTitle(df,map_means)
    titles_dict = {'Capt': 'Other',
               'Major': 'Other',
               'Jonkheer': 'Other',
               'Don': 'Other',
               'Sir': 'Other',
               'Dr': 'Other',
               'Rev': 'Other',
               'Countess': 'Other',
               'Dona': 'Other',
               'Mme': 'Mrs',
               'Mlle': 'Miss',
               'Ms': 'Miss',
               'Mr': 'Mr',
               'Mrs': 'Mrs',
               'Miss': 'Miss',
               'Master': 'Master',
               'Lady': 'Other'}
    
    df['Title']=df['Title'].map(titles_dict)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','PassengerId'], axis=1, inplace=True)
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    fsize_scale_param=scaler.fit(df['FamilySize'])
    df['Fsize_scaled'] = scaler.fit_transform(df['FamilySize'], fsize_scale_param)
    df.drop(['Age','Fare','FamilySize','Sex_male','Pclass_2','Embarked_S','Cabin_No'],axis=1,inplace=True,errors='ignore')
    dummies_Title=pd.get_dummies(df['Title'],prefix='Title')
    df=pd.concat([df,dummies_Title],axis=1)
    df.drop(['Title','Title_Other'],axis=1,inplace=True)
    return df

def train_phase():
    #data_train1, rfr = set_missing_ages(data_train)
    train_np = df.as_matrix()
    y = train_np[:, 0]
    X = train_np[:, 1:]
    
    cList=np.linspace(0.2,1.0,5)
    bestScore=0
    bestC=0
    for c in cList:
        clf = linear_model.LogisticRegression(C=c, penalty='l2', tol=1e-6)
        clf.fit(X,y)
        scores=cross_validation.cross_val_score(clf, X, y, cv=10)
        print (scores)
        currentScore=np.mean(scores)
        if currentScore>bestScore :
            bestScore=currentScore
            bestC=c
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    clf = linear_model.LogisticRegression(C=bestC, penalty='l2', tol=1e-6)
    clf = clf.fit(X,y)
    plot_learning_curve(clf, u"learning curve", X, y,cv=5)
    print(pd.DataFrame({"columns":list(df.columns)[1:], "coef":list(clf.coef_.T)}))
    return clf

def set_Title(df):
    df['Title']=0
      
    for i in df:
        df['Title']=df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
    return df
        

# 可以试着加 GridSearchCV来找 hyper-params 
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        try :
            print(self.clf.fit(x,y).feature_importances_)
            return self.clf.fit(x,y).feature_importances_
        except :
            print('This model does not have feature_importances_')
        
    def cv_score(self,x,y,cv=5):
        scores=cross_validation.cross_val_score(self.clf.fit(x,y), x, y, cv=cv)
        return np.mean(scores)
    
    def score(self,x,y):
        return self.clf.fit(x,y).score(x,y)
                

data_train = pd.read_csv("/Users/douglas/kaggle titanic/train.csv")
df=data_train.copy()
df=set_Title(df)
means = df.groupby('Title')['Age'].mean()
map_means = means.to_dict()
df=process_data(df,map_means)
train_np = df.as_matrix()
ytest = train_np[0:100, 0]
Xtest = train_np[0:100, 1:]
X=train_np[101:, 1:]
y=train_np[101:,0]
#pearson graph

#colormap = plt.cm.RdBu
#plt.figure(figsize=(14,12))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
#sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#            square=True, cmap=colormap, linecolor='white', annot=True)


#random forest

rf_params = {
    'n_jobs': -1,
    'n_estimators': 30,
    'warm_start': False, 
    #'max_features': 0.2,
    'max_depth': 6,
    #'min_samples_leaf': 2,
    'max_features' : 'auto',
    'oob_score':True,
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':30,
    'max_features': 'auto',
    'max_depth': 6,
    'verbose': 0,
    'bootstrap':True
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 1,
    #'base_estimator':RandomForestClassifier()
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 20,
     'max_features': 0.2,
    'max_depth': 6,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'rbf',
    'C' : 2,
}

log_params={
    'C':1,
    'penalty':'l2',
    'tol':1e-6
}

nn_params = {
    'solver':'adam',
    'activation':'tanh',
    'learning_rate':'adaptive',
    'hidden_layer_sizes':(8,)
}    

SEED=1
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
log = SklearnHelper(clf=linear_model.LogisticRegression, seed=SEED, params=log_params)
nn=SklearnHelper(clf=MLPClassifier,seed=SEED,params=nn_params)

classifiers=[rf,et,ada,gb,log,svc,nn]
#classifiers=[rf,et,ada,gb]
#train_prediction=df.copy()
train_prediction=pd.DataFrame()


for c in classifiers:
    c.train(X,y)
    #print(c.score(X,y))
    clf_name=type(c.clf).__name__
    #plot_learning_curve(c.clf, clf_name+" "+"learning curve", X, y,cv=3)
    prediction=c.predict(X)
    train_prediction[clf_name]=prediction
    print(clf_name+" score: "+ str(c.score(X,y)))
    print(clf_name+" cv score: "+str(c.cv_score(X,y)))
    
train_np2=train_prediction.as_matrix() 
X2=train_np2
n,d=X2.shape
temp=[]
for i in range(0,n):
    if np.count_nonzero(X2[i,:])>(d/2):
        temp.append(1)
    else:
        temp.append(0)
temp=np.array(temp)
a=temp==y
np.count_nonzero(a)/n 
    

gb_params2 = {
    'n_estimators': 100,
    'max_features': 'auto',
    'max_depth': 10,
    'verbose': 0
}


                
gb2=SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params2)   
gb2.train(X2,y)
print(gb2.score(X2,y))
print(gb2.cv_score(X2,y))


rf_params2 = {
    'n_jobs': -1,
    'n_estimators': 30,
    'warm_start': False, 
    #'max_features': 0.2,
    'max_depth': 6,
    #'min_samples_leaf': 2,
    'max_features' : 'auto',
    'oob_score':True,
    'verbose': 0
}

rf2=SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params2)   
rf2.train(X2,y)
print(rf2.score(X2,y))
print(rf2.cv_score(X2,y))


local_test = pd.DataFrame()
for c in classifiers:
    clf_name=type(c.clf).__name__
    prediction_Xtest=c.predict(Xtest)
    local_test[clf_name]=prediction_Xtest
  
nn_params = {
    'solver':'lbfgs',
    'activation':'tanh',
    'learning_rate':'adaptive',
    'hidden_layer_sizes':(100,)
}    
nn=SklearnHelper(clf=MLPClassifier,seed=SEED,params=nn_params)
nn.train(X2,y)
print(nn.score(X2,y))
print(nn.cv_score(X2,y))            
local_predict_nn=nn.predict(local_test)
np.count_nonzero(local_predict_nn==ytest)              
               
local_predict_gb2=gb2.predict(local_test)
np.count_nonzero(local_predict_gb2==ytest)

log2=SklearnHelper(clf=linear_model.LogisticRegression, seed=SEED, params=log_params)   
log2.train(X2,y)
print(log2.score(X2,y))
print(log2.cv_score(X2,y))

local_predict_log=log2.predict(local_test)
np.count_nonzero(local_predict_log==ytest)    

data_test = pd.read_csv("/Users/douglas/kaggle titanic/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
                           
df_test=set_Title(data_test)
df_test=process_data(data_test,map_means)


#df_test_exp=df_test.copy()
df_test_exp=pd.DataFrame()

for c in classifiers:
    clf_name=type(c.clf).__name__
    prediction=c.predict(df_test)
    df_test_exp[clf_name]=prediction
    #print(pd.DataFrame({"columns":list(df.columns)[1:], "coef":gb.feature_importances(X,y)}))
      
Xtest=df_test_exp.as_matrix()
stack_prediction2=gb2.predict(Xtest) 
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':stack_prediction2.astype(np.int32)})
result.to_csv("/Users/douglas/kaggle titanic/stack_predictions2.csv", index=False)

   
stack_prediction=[]         
Xtest=np.asmatrix(df_test_exp)
n,d=Xtest.shape
for i in range(0,n):
    if np.count_nonzero(Xtest[i,:])>2:
        stack_prediction.append(1)
    else:
        stack_prediction.append(0)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':stack_prediction})
result.to_csv("/Users/douglas/kaggle titanic/stack_predictions.csv", index=False)




predictions=logclf.predict(df_test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/douglas/kaggle titanic/logistic_regression_predictions.csv", index=False)




