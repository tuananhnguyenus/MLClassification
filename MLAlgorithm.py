import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import tree
from keras_self_attention import SeqSelfAttention
import pickle
import random
import numpy as np
from joblib import dump, load
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pathlib



class MLClassification:
    def __init__(self,name):    
        self.name= name
        self.deeplearning={"rnn","lstm"}
        
        self.saved_model = os.path.join(pathlib.Path().absolute(),"model_"+name)
        if not os.path.exists(self.saved_model) and self.name in self.deeplearning: 
            os.mkdir(self.saved_model)
        

    def create_model(self,dim=None): pass

    def prepare_train_test_data(self,csv_file,csv_header=None):
        self.dataset = pandas.read_csv(csv_file, header=csv_header).values
               
        self._num_features = len(self.dataset[0])
        self.X = self.dataset[:,0:self._num_features-1]
        self.Y = self.dataset[:,self._num_features-1]
        
        encoder = LabelEncoder()        
        encoder.fit(self.Y)

        self.Y = encoder.transform(self.Y)
        self.categoryY = np_utils.to_categorical(self.Y)
        
        self.X_train,self.X_test,self.Y_train,self.Y_test =train_test_split(self.X,self.categoryY,train_size=0.7)
        self.num_clusters = len(self.categoryY[0])

    def train(self,csv_file,csv_header=None,outfile=""):
        self.prepare_train_test_data(csv_file)

        self.create_model()
        
        if self.name == 'lstm':
            self.X_train=self.X_train.reshape(self.X_train.shape[0],1,self.X_train.shape[1])
            self.X_test = self.X_test.reshape(self.X_test.shape[0],1,self.X_test.shape[1])
            self.X= self.X.reshape(self.X.shape[0],1,self.X.shape[1])

        if self.name in self.deeplearning:
             self.model.fit(self.X_train,self.Y_train, epochs=10,batch_size=5,verbose=1)
             self.model.save(self.saved_model)
        else:  
            self.model.fit(self.X_train,self.Y_train)
        
            fw = open(self.saved_model,'w')
            dump(self.model, self.saved_model)
            fw.close()
        
        y_train_pred = self.model.predict(self.X_train)
        train_error = mean_squared_error(y_train_pred,self.Y_train)
        print("model name {}, train_error {}".format(self.name,train_error))

        y_test_pred= self.model.predict(self.X_test)
        print("model name {}, test_error {}".format(self.name,mean_squared_error(y_test_pred,self.Y_test)))
        
        y_pred = self.model.predict(self.X)
        outfile+=self.name
        pandas.DataFrame(y_pred).to_csv(outfile)
        
        r=[]
        for yy in y_pred:
            r.append(np.argmax(yy))
        pandas.DataFrame(r).to_csv("train_predicted_"+outfile)

    def predict(self,in_csv_file,out_csv_file=None,csv_header=None):
        if os.path.exists(self.saved_model):
            if self.name in self.deeplearning:
                self.model =tf.keras.models.load_model(self.saved_model)
                    
            else:
                self.model=load(self.saved_model)
        else: return "No model found"

        self.dataset = pandas.read_csv(in_csv_file, header=csv_header).values
        if self.name == 'lstm':
            self.dataset= self.dataset.reshape(self.dataset.shape[0],1,self.dataset.shape[1])

        y = self.model.predict(self.dataset)
        r=[]
        for yy in y:
            r.append(np.argmax(yy))
        if out_csv_file:
            pandas.DataFrame(r).to_csv(outfile)
        else:
            pandas.DataFrame(r).to_csv('predicted_'+self.name)

class RNNAlg(MLClassification):
    def __init__(self):   
        super().__init__("rnn")
        
    def create_model(self):
        self.model = Sequential()
        
        self.model.add(Dense(10,input_dim=self.X.shape[1],activation='relu'))
        self.model.add(Dense(self.num_clusters,activation='softmax'))        
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


class RandomForestAlg(MLClassification):
    def __init__(self):         
        super().__init__("random_forest")

    def create_model(self):
        self.model=RandomForestClassifier(n_estimators=100)
        
class DecisionTreeAlg(MLClassification):
    def __init__(self):         
        super().__init__("decision_tree")

    def create_model(self):
        self.model=tree.DecisionTreeClassifier()
        
class KmeansAlg(MLClassification):
    def __init__(self):         
        super().__init__("kmeans")

    def train(self,training_csv_file,outfile=""):
        self.prepare_train_test_data(training_csv_file)
        self.__kmeans5 = KMeans(n_clusters=self.num_clusters)

        y_pred = self.__kmeans5.fit_predict(self.X)
        dump(self.__kmeans5, self.saved_model)
        outfile+=self.name
        pandas.DataFrame(y_pred).to_csv(outfile)

    def predict(self,in_csv_file,outfile=None,csv_header=None):
        if (os.path.exists(self.saved_model)):
            self.model=load(self.saved_model)
        self.dataset = pandas.read_csv(in_csv_file, header=csv_header).values
        y_pred = self.__kmeans5.fit_predict(self.dataset)
        if outfile: pandas.DataFrame(y_pred).to_csv(outfile)

class LSTMAlg(MLClassification):
    def __init__(self):   
        super().__init__("lstm")
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(10,input_shape=(1,self.X.shape[1]),activation='relu'))
        self.model.add(Dense(self.num_clusters,activation='softmax'))        
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    
        
           

