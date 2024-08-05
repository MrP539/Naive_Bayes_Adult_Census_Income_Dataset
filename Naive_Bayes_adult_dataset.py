import sklearn
import sklearn.datasets
import pandas as pd
import numpy as np
import os

import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.preprocessing
from tool import find_nan_values_at_column

from tool import create_confusion_matrix


raw_data = pd.read_csv(os.path.join(r"D:\machine_learning_AI_Builders\ML_Algorithm\Naive_Bayes_Adult_Census_Income_Dataset\data\adult.csv"))

print(raw_data)

print(raw_data.columns.values.tolist())



find_nan_values_at_column.check_type_and_any_values(raw_data.columns.tolist(),raw_data,"?")

######################## ecode oject --> numeric ################################

# ฟังก์ชัน encode พร้อมเก็บข้อมูลการแปลงค่า
def encode(data_df):
    label_encoders = {}  # สร้าง dictionary เพื่อเก็บ LabelEncoder ของแต่ละ column
    for column in data_df.columns:
        if data_df[column].dtype == type(object):
            label_encoder = sklearn.preprocessing.LabelEncoder()
            encoded_values = label_encoder.fit_transform(data_df[column])
            data_df[column] = encoded_values
            label_encoders[column] = label_encoder  # เก็บ LabelEncoder ไว้ใน dictionary
    return data_df, label_encoders

# เรียกใช้ฟังก์ชัน encode
data_encoded, label_encoders = encode(raw_data)

# แสดงข้อมูลหลังจาก encode
print(data_encoded)

# ตรวจสอบ mapping ของค่าเดิมกับค่าที่ถูก encode ในแต่ละ column
encode_desception={k:{original_value:encoded_value for original_value, encoded_value in zip(values.classes_, range(len(values.classes_)))}   for k,values in label_encoders.items() }

# for column, label_encoder in label_encoders.items():
    
#     print(f"\nMapping for column {column}:")
#     for original_value, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):

#         encode_desception[column] ={original_value:encoded_value}
#         print(f"{original_value} -> {encoded_value}")

print("\n----")
print(encode_desception["income"]["<=50K"])
print("\n----")


feature = data_encoded.drop(labels="income",axis=1)
targets = data_encoded["income"]

print(feature,targets)

x_train,x_test,y_train,y_test =sklearn.model_selection.train_test_split(feature,targets,random_state=42,test_size=0.2,shuffle=True)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


NB_model = sklearn.naive_bayes.GaussianNB()
NB_model.fit(X=x_train,y=y_train)

pred = NB_model.predict(x_test)

print(pred)

print(f"evaluate : Accuracy == {sklearn.metrics.accuracy_score(y_true=y_test,y_pred=pred)}")
print(f"evaluate : Accuracy == {NB_model.score(x_test,y_test)}")

print(sklearn.metrics.classification_report(y_true=y_test,y_pred=pred,target_names=["<=50K",">50K"]))
create_confusion_matrix.CREATE_CONFUSION_MATRICS(y_actual=y_test,y_pred=pred,numclass=len(set(targets.values))-1)