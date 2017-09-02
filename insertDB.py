import heapq
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint 
from sklearn import datasets
import pymongo
from pymongo import MongoClient
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture

transmitters = 0
gridResolution = 0
totalGrid = 0
partition = 0

def getCentroid_totalarea(x,y):
        centX = y*gridResolution + float(gridResolution)/2
        centY = x*gridResolution + float(gridResolution)/2
        return float(centX),float(centY)

def getCords(string):
	li = string.split(",")
	return int(li[0]),int(li[1])
	
    
def hashed(x,y,subArea,m):
	m_x = subArea* (m %partition ) 
	m_y = subArea* (m / partition ) 
	res = ''
	res = res + str(m_x + y*gridResolution) + "," + str(m_y + x*gridResolution)
	return res

def getCentroid(x,y,subArea,m):
	m_x = m/partition
	m_y = m % partition
	centX = m_y * subArea + y*gridResolution + float(gridResolution)/2
	centY = m_x * subArea + x*gridResolution + float(gridResolution)/2
	return float(centX),float(centY)

def distance(x,y,tloc):
	#x,y = getCentroid(x,y)
	#print tloc[0],tloc[1]
	d = math.sqrt(float((x-tloc[0])*(x-tloc[0]) + (y-tloc[1])*(y-tloc[1])))
	return d
	
def generateREM(transLoc,sd,tIndex,subArea,model):
	data = []
	for t in range(transmitters):
		dict = {}
		for x in range(0,subArea/gridResolution):
			for y in range(0,subArea/gridResolution):
				c1,c2 = getCentroid(x,y,subArea,model)
			     	if(c1 == transLoc[t+tIndex][0] and c2 == transLoc[t+tIndex][1]):
					continue
				vals = []
				for dev in sd:
					rssi = 10*math.log10(16) - math.log10(distance(c1,c2,transLoc[t+tIndex]))*10*4 - dev 
					vals.append(rssi)
				dict[hashed(x,y,subArea,model)] = vals
		data.append(dict)
	#print data
	return data

def mean(a):
    return sum(a) / len(a)

def generateVectors(transLoc,data,tIndex,subArea,model):
	vec = {}
	init_mean = []
	for x in range(0,subArea/gridResolution):
		for y in range(0,subArea/gridResolution):
			vals = []	
			for i in range(0,20):
				vector = []
				trInd = 0 + tIndex
				for t in data:
			 		c1,c2 = getCentroid(x,y,subArea,model)
					if(c1 == transLoc[trInd][0] and c2 == transLoc[trInd][1]):
						vector.append(10*math.log10(16))
					else:
						vector.append(t[hashed(x,y,subArea,model)][i])	
					trInd = trInd + 1
				vals.append(vector)
			average =  [sum(items) / len(vals) for items in zip(*vals)]
			init_mean.append(average)	
			vec[hashed(x,y,subArea,model)] = vals
	#print vec
	return vec,init_mean

def get_db():
    client = MongoClient('localhost:27017')
    db = client.locp
    return db

def addtoDBmap(transLoc,map,db):
	for k,v,model in zip(transLoc.keys(),transLoc.values(),map):
                db.modelMap.insert({"transmitter" : k,"tloc" : v,"model" : model})

def addtoDB(train_data,test_data,train_labels,test_labels,init_mean,db):
	for k in range(0,len(train_data)):
		db.locationPredTrain.insert({"vector" : train_data[k],"label" : train_labels[k]})
	for k in range(0,len(test_data)):
		db.locationPredTest.insert({"vector" : test_data[k] , "label" : test_labels[k]})
	for k in range(0,len(init_mean)):
		db.initMean.insert({"vector" : init_mean[k]})

def getDB(db):
	return db.locationPred.find_one()

def transmitterLocs(transLoc,count,randomGrids):
        k = 1000/gridResolution
        x_coords = []
        y_coords = []
        while count > 0:
           if count < k:
                x_coords = x_coords + random.sample(xrange(k),count)
                y_coords = y_coords + random.sample(xrange(k),count)
                count = 0
           else:
                x_coords = x_coords + random.sample(xrange(k),k)
                y_coords = y_coords + random.sample(xrange(k),k)
                count = count - k
        ind = 0
        for x,y in zip(x_coords,y_coords):
                cords = []
                cx,cy = getCentroid_totalarea(x,y)
                cords.append(cx)
                cords.append(cy)
                transLoc[ind] = cords
                ind = ind + 1

def transmitterLoc(transLoc,subArea,randomGrids):
	n = 0	
	while ( n < len(randomGrids)):
		model = randomGrids[n]
		x_coords = random.randrange(0,subArea/gridResolution)
                y_coords = random.randrange(0,subArea/gridResolution)
                cords = []
                cx,cy = getCentroid(x_coords,y_coords,subArea,model)
                cords.append(cx)
                cords.append(cy)
                transLoc[n] = cords
		n = n + 1


def generateData(vectors,subArea):
	train_data = []
	train_labels = []
	test_labels = []
	randints = []
	test_data = []
	totalVec = []
	totalClasses = []
	for key in vectors.keys():
		for vec in vectors[key]:
			totalClasses.append(key) 
			totalVec.append(vec)
	#k = 5 * (subArea/gridResolution) * (subArea/gridResolution)
	k = 0.2*len(totalVec)
	k = int(k)
	randints = random.sample(xrange(len(totalVec)),k)
	for i in range(len(totalVec)):
		if i in randints:
			test_data.append(totalVec[i]) 
			test_labels.append(totalClasses[i]) 
		else:
			train_data.append(totalVec[i])
			train_labels.append(totalClasses[i])
	return train_data,train_labels,test_data,test_labels


def generateMap(transLoc,partition,subArea):
	map = []
	for t in transLoc.values():
		minDist = 999999
		for model in range(partition*partition): 
			x = (model%partition)*subArea + float(subArea/2)  
			y = (model/partition)*subArea + float(subArea/2)
			d = distance(x,y,t)
			if d < minDist:
				minDist = d
				best_model = model
		map.append(best_model)
		print "*****************"
	return map
			
		
if __name__ == '__main__':

    noTransmitters = []
    gridRes = []
    area = []
    if len(sys.argv) == 5:
        noTransmitters.append(int(sys.argv[1]))
        gridRes.append(int(sys.argv[2]))
        area.append(int(sys.argv[3]))
        partition = (int(sys.argv[4]))
    elif len(sys.argv) == 1:
    	noTransmitters = [3,3,8,8]
    	gridRes = [5,5,5,5]
    db = get_db()
    for transmitters,gridResolution,totalGrid in zip(noTransmitters,gridRes,area):
    		subArea = totalGrid/partition
		transLoc = {}
		randomGrids = random.sample(xrange(partition*partition),transmitters)
		#transmitterLoc(transLoc,subArea,randomGrids)
		transLoc = {0: [670.0, 10.0], 1: [870.0, 890.0], 2: [250.0, 370.0], 3: [130.0, 190.0], 4: [330.0, 570.0], 5: [990.0, 90.0], 6: [410.0, 710.0], 7: [490.0, 950.0], 8: [470.0, 230.0], 9: [490.0, 490.0]}
		print transLoc
		map = generateMap(transLoc,partition,subArea)
		addtoDBmap(transLoc,map,db)
		sd = []
		for r in range(0,20):
			randNoise = np.random.normal(0,4)
			sd.append(randNoise)
		for m in range(partition*partition):
			data = generateREM(transLoc,sd,0,subArea,m)
			vectors,init_mean = generateVectors(transLoc,data,0,subArea,m)
			train,classes,test,expectedLoc = generateData(vectors,subArea)
#			print train 
#			print classes 
#			print test
			addtoDB(train,test,classes,expectedLoc,init_mean,db)
		
