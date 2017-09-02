import heapq
import sys
import time
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from random import randint
from sklearn import datasets
import pymongo
from pymongo import MongoClient
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture 

partition =0

def getDataInBatches(cursor,batch_size,attr):
    count=0
    max_count=batch_size
    batch = []
    while count<max_count :
        count+=1
        try:
            record = cursor.next()
            batch.append(record.get(str(attr)))
        except StopIteration:
            print("Empty cursor!")
    return batch

def get_db():
    client = MongoClient('localhost:27017')
    db = client.locp
    return db

if __name__ == '__main__':
		gridResolution = int(sys.argv[1])
		area = int(sys.argv[2])
		partition = int(sys.argv[3])
		k = 0
		go =0
		db = get_db()
		cursor_train=db.locationPredTrain.find(timeout=False)
		cursor_train_label=db.locationPredTrain.find(timeout=False)
		n_train = db.locationPredTrain.find().count()
		subarea=area/partition
		batch_size = (subarea/gridResolution)*(subarea/gridResolution)

		cursor_means=db.initMean.find(timeout = False)			
		
 		
		models=[]
		print("model made")
		count=0
		while k < n_train:
				count+=1
				init_mean = getDataInBatches(cursor_means, batch_size , "vector")
				init_mean = np.array(init_mean)
				gmm = mixture.GaussianMixture(n_components=batch_size, means_init=init_mean ,covariance_type='tied', warm_start=False)
				train = getDataInBatches(cursor_train,batch_size*20*0.8,"vector")
				k = k + batch_size*20*0.8
				train = np.array(train)
				gmm.fit(train)
				models.append(gmm)				
				print "*****************************************"
				
		print("training done")
		with open('model_.pickle','wb') as f:
				pickle.dump(models,f)
