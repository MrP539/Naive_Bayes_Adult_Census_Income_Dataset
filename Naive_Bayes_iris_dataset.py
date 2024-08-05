import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import Naive_Bayes.create_confusion_matrix as create_confusion_matrix

raw_data = sklearn.datasets.load_iris()

print(raw_data.keys())

feature = raw_data["data"]
targets = raw_data["target"]

print(feature.shape,targets.shape)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(feature,targets,test_size=0.2,shuffle=True,random_state=101)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
print(y_train.shape,y_test.shape)

NB_model = sklearn.naive_bayes.GaussianNB()
NB_model.fit(X=x_train,y=y_train)

pred = NB_model.predict(x_test)

print(pred)

print(f"evaluate : Accuracy == {sklearn.metrics.accuracy_score(y_true=y_test,y_pred=pred)}")
print(f"evaluate : Accuracy == {NB_model.score(x_test,y_test)}")

print(sklearn.metrics.classification_report(y_true=y_test,y_pred=pred,target_names=raw_data["target_names"]))

