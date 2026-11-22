from sklearn.datasets import load_iris
import pandas as pd

#load iris dataset
iris=load_iris()

#Convert Dataset into Dataframe
df=pd.DataFrame(iris.data,columns=iris.feature_names)

#Add species column
df['Species']=iris.target

#convert number into species names
df['Species'] = df['Species'].map({0:'Setosa',1:'Versicolor',2:'Virginica'})

#Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

#check Duplicate Values
print("Duplicate Row:")
print(df.duplicated().sum())

#Statistical Information 
print("Statistical Summary:")
print(df.describe())

#Dataset information
print("Dataset Shape:")
print(df.info())

import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure(figsize=(6,4))

sns.countplot(data=df,x='Species')

plt.title("Count of Each Iris Species")
plt.xlabel("Flower Species")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(7,4))

sns.boxplot(data=df,x='Species',y='petal length (cm)')
plt.title("Petal Length of Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal LEngth(cm)")
plt.show()

from sklearn.model_selection import train_test_split
#Select Features 
X=iris.data
#Select Target 
y=iris.target
#Split dataset
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
#Print Dataset Size
print("Training Data:",X_train.shape)
print("Testing Data:",X_test.shape)
print("Training Target:",y_train.shape)
print("Testing Target:",y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
#Create KNN Model
model=KNeighborsClassifier(n_neighbors=3)
#Train the Model
model.fit(X_train,y_train)
print("Model Training Completed!")

#Preedict Test Data
y_pred = model.predict(X_test)
print("Predicted Values:")
print("Actual Values:")
print(y_test)

from sklearn.metrics import accuracy_score
#Calculate Accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Model Accuracy:",accuracy*100,"%")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#confusinon Matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
#Classification Report
print("Classification Report:")
print(classification_report(y_test,y_pred,target_names=iris.target_names))
#Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",xticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicated")
plt.ylabel("Actual")

plt.show()

#New Flower Prediction 
new_flower = [[5.1 , 3.5 , 1.4 , 0.2]]
#Predict flower species
prediction=model.predict(new_flower) 
#Get species Name
flower_name=iris.target_names[prediction[0]]
print("Prediction Flower Species:",flower_name)

import joblib
#save Model
joblib.dump(model,"iris_model.pkl")
print("Model Saved Successfully!")