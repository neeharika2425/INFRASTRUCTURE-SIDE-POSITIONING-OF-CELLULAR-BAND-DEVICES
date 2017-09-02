import heapq
import sys
import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint
from sklearn import datasets
import pymongo
from pymongo import MongoClient
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture 
import localization as lx
import operator

partition=0
gridResolution = 0
subarea = 0

def get_db():
    client = MongoClient('localhost:27017')
    db = client.locp
    return db

def distance(x,y,tloc):
        #x,y = getCentroid(x,y)
        #print tloc[0],tloc[1]
        d = math.sqrt(float((x-tloc[0])*(x-tloc[0]) + (y-tloc[1])*(y-tloc[1])))
        return d

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

def getLocation(x,y,m):
 	m_x = subarea* (m %partition )
        m_y = subarea* (m / partition )
        res_x  = m_x + y*gridResolution
	res_y = m_y + x*gridResolution
        return res_x,res_y

def getCords(string):
	li = string.split(",")
	return float(li[0]),float(li[1])


def calculateLocError(observed,expected):
	err = []
	for o,e in zip(observed,expected):
		x1,y1 = getCords(o)
		x2,y2 = getCords(e)
		#x1,y1 = getCentroid(x1,y1)
		#x2,y2 = getCentroid(x2,y2)
		d = math.sqrt(float((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)))
		err.append(d)
			
	return err

def convertToString(labels,n):
    result = []
    for i in labels:
    	x = i/n
    	y = i%n
    	result.append( str(x) + ","+ str(y))
    return result


def predictFromProb(log_prob,models):
	locations = []
	n = subarea/gridResolution
	for vector,model in zip(log_prob,models):
		loc_x = 0
		loc_y = 0
		for i in range(len(vector)):
			x,y = getLocation(i%n,i/n,model)
			loc_x = loc_x + vector[i]*x
			loc_y = loc_y + vector[i]*y
		locations.append(str(loc_x) + "," + str(loc_y))
	return locations
def dist(rssi):
	dist = math.pow(10,((10*math.log10(16) - rssi - 4 )/4*10))	
	return dist

def predictModel(locs,trans,test):
	P=lx.Project(mode='2D',solver='LSE')
	
	P.add_anchor('t1', tuple(locs[0]))
	P.add_anchor('t2',tuple(locs[1]))
	P.add_anchor('t3',tuple(locs[2]))

	target,label = P.add_target()
	target.add_measure('t1',dist(test[trans[0]]))
	target.add_measure('t2',dist(test[trans[1]]))
	target.add_measure('t3',dist(test[trans[2]]))

	P.solve()
	t = []		
	t.append(target.loc.x)
	t.append(target.loc.y)
	minDist = 999999
	for model in range(partition*partition):
                        x = (model%partition)*subarea + float(subarea/2)
                        y = (model/partition)*subarea + float(subarea/2)
                        d = distance(x,y,t)
                        if d < minDist:
                                minDist = d
                                best_model = model
	return best_model

if __name__ == '__main__':

	gridResolution = int(sys.argv[1])
	area = int(sys.argv[2])	
	partition = int(sys.argv[3])
	flag = int(sys.argv[4])
	subarea = area/partition
	db = get_db()
	models=[]
	with open('model_.pickle','rb') as f:
		models = pickle.load(f)

	for model in models:
		print(model)
		print("------------------")
	
	cursor_test=db.locationPredTest.find()
	cursor_test_label=db.locationPredTest.find()
	cursor_map=db.modelMap.find()
	n_test = db.locationPredTest.find().count()
		
	tests = getDataInBatches(cursor_test, n_test,"vector")
	tests = np.array(tests)
	expectedLoc = getDataInBatches(cursor_test_label, n_test ,"label")	
	

	if flag == 0:
		log_prob=[]
		model_chosen = []
		for test in tests:
			test_vec = []
			arr = np.array(test)
			top_trans = arr.argsort()[-3:][::-1]
			loc_t = []
			for i in top_trans:
				info = db.modelMap.find_one({ "transmitter" : i})
				loc_t.append(info['tloc'])
			model_num = predictModel(loc_t,top_trans,test)
			test_vec.append(test)	
			log_prob.append(models[model_num].predict_proba(test_vec)[0])
	 		model_chosen.append(model_num)
		print model_chosen
		pred_labels  = predictFromProb(log_prob,model_chosen)	
		#print pred_labels
		#print "*****"
		#print expectedLoc
		err = calculateLocError(pred_labels,expectedLoc)		
		#print err
		print "Mean localization error is"
		print np.mean(err)
	if flag == 1:
		size = [0,0.1,0.3,0.4,0.5,0.6]
		semi_error = []
		anchors = []
		i = 0
		for s in size:
			log_prob=[]
			model_chosen = []
			semi = int(len(tests)*s)
			anchors.append(semi)
			pred = []
			for i in range(semi):
				pred.append(expectedLoc[i])
			for i in xrange(semi,len(tests)):
				test_vec = []
				arr = np.array(tests[i])
				top_trans = arr.argsort()[-3:][::-1]
				loc_t = []
				for i in top_trans:
					info = db.modelMap.find_one({ "transmitter" : i})
					loc_t.append(info['tloc'])
				model_num = predictModel(loc_t,top_trans,tests[i])
				test_vec.append(tests[i])	
				log_prob.append(models[model_num].predict_proba(test_vec)[0])
	 			model_chosen.append(model_num)
			pred_labels  = predictFromProb(log_prob,model_chosen)	
			pred = pred + pred_labels
			err = calculateLocError(pred,expectedLoc)		
			print "Mean localization error is"
			print np.mean(err)
			semi_error.append(np.mean(err))
		plt.plot(anchors,semi_error)
		plt.title('Localization error with increase in labeled data')
		plt.ylabel('Localization Error')
                plt.xlabel('Training data with labeled data')
		plt.show()



	
	
