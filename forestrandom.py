import pandas as pd
import numpy as np
import collections
from collections import deque
import copy
import math
import matplotlib.pyplot as plt
from matplotlib import style
from random import randrange
from PIL import Image
from collections import Counter
import pickle
from node import Node



  

 ## The start of creating trees in the random forest passed in are a matrix and the corresponding classification labels/headers    
def tree(matrix,class_labels):
    df=pd.DataFrame(data=matrix, columns = class_labels)
## starting the insert method to to insert data into trees- this method calls all the other necessary functions
    head=insert(df)
    return head
## this method inserts data into the tree models 
def insert(df):
        
    try:
        ## find which classifier to split on and pass as a dictionary is the format of {"name":df.columns.values[i],"00":[],"01":[],"10":[],"11":[],"gain":0}
        temp_dict=split(df)
        node=Node(temp_dict["name"])
        ## set entropy level
        node.setEntropy(temp_dict["gain"])
        ## the classifier which we are splitting needs to be dropped since we are splitting on this classifier
        df.drop(temp_dict["name"], axis=1, inplace=True)
        node_right=Node(None)
        node_left=Node(None)
        ## two parts to splitting if columns <2 then we need still have other columsn will only end a side of the tree if 100% complete 
        if len(df.columns)>2:
            ## For a negative output of a classifier does it correspond to either not pure meaning not 100% yes or not 100% no or 0% for each meaning not found then we do not end the left
        ## side of the tree but split via  the next classifier
            if (len(temp_dict["00"])>0 and len(temp_dict["01"])>0) or ((len(temp_dict["00"])==0) and (len(temp_dict["01"])==0)):
                left_list=[]
                left_list.extend(temp_dict["00"])
                left_list.extend(temp_dict["01"])
                df_left=pd.DataFrame(columns= list(df.columns.values))
                for i in left_list:
                    df_left=df_left.append(df.iloc[[i]],ignore_index=True)
                print("hello",df_left)
        ## as described in the recent previous comment we take the data from the 0's which will be sent to the left node to continue the split
                node.setLeft(insert(df_left))
                ## in this else we know that the data is pure meaning if 0 for the classifier category it is 100% yes or no for the dependant classifier(independant varialble
                ## if it is all 0's in indpendant variable then we set the node left to zero and end the recursion if we have all 1's then the left is set to node 1 and end recursion
            else:
                if len(temp_dict["00"])>len(temp_dict["01"]):
                    node.setLeft(Node(0))
                else:
                    node.setLeft(Node(1))
        ## for a classifier being one we follow same pattern as above but this time everything moves to node.right() as opposed to left
            if(len(temp_dict["10"])>0 and len(temp_dict["11"])>0)or ((len(temp_dict["10"])==0) and (len(temp_dict["11"])==0)):
                right_list=[]
                right_list.extend(temp_dict["10"])
                right_list.extend(temp_dict["11"])
                df_right=pd.DataFrame( columns = list(df.columns.values))
                for i in right_list:
                    df_right=df_right.append(df.iloc[[i]],ignore_index=True)
                node.setRight(insert(df_right))
            else:
                if len(temp_dict["10"])>len(temp_dict["11"]):
                    node.setRight(Node(0))
                else:
                    node.setRight(Node(1))
        ## if length of columns less than one purity does not matter the tree ends here, if classifier is 1 we set right to 1 if dependant var is 1 or 0 if dependant var is 0
        elif len(df.columns)>1:
            if len(temp_dict["10"])>len(temp_dict["11"]):
                node.setRight(Node(0))
            else:
                node.setRight(Node(1))
        ## same for left side but not set right but we set left
            if len(temp_dict["00"])>len(temp_dict["01"]):
                node.setLeft(Node(0))
            else:
                node.setLeft(Node(1))
        ## we found a small issue and were unable to fully debug it, if this occured we ignore that trial to split and end the recursion as to not create more problems
    except:
        pass
        print("ERROR IGNORE THIS TRIAL")
        
    return node

## these two are just used to ensure non math errors because if a class has 0 of a 1 or 0 then the probobility is 0/0 so this prevents that and just returns 0
def check_zero_prob(numer,divis):
    numb_ret=0
    try:
        numb_ret=float(numer/divis)
    except:
        numb_ret=numb_ret
    return(numb_ret)

## because entropy is calculated using logs in the chance that the proboility passed into the log is 0 we need to just return 0 because log 0 is impossible note- even if taking the limit
## entropy is probability log(probability) and the probability of 0 cancels the log of 0  based on L'hopitals 
def check_for_zero_entropy(prob0,prob1):
    store_numbers=[prob0,prob1]
    total_entropy=0
    for i in store_numbers:
        try:
            total_entropy=total_entropy+float(-i*math.log(i,2))
        except:
            total_entropy=total_entropy
    return (total_entropy)

## calculates entropy of data
def calc_entropy(dictionary,last):
    # these variables just for ease of readability- should be self explanatory
        total0=float(len(dictionary["00"]))+float(len(dictionary["01"]))
        probability00=check_zero_prob(float(len(dictionary["00"])),total0)
        probability01= check_zero_prob(float(len(dictionary["01"])),total0)
        total1=float(len(dictionary["10"]))+ float (len(dictionary["11"]))
        probability10=check_zero_prob(float(len(dictionary["10"])),total1)
        probability11=0 if total1==0 else float (len(dictionary["11"]))/float(total1)
        spec_cond_entropy0=check_for_zero_entropy(probability00,probability01)
        spec_cond_entropy1=check_for_zero_entropy(probability10,probability11)
        grandtotal=total0+total1
        probability0=(float(len(dictionary["00"]))+float(len(dictionary["01"])))/grandtotal
        probability1=(float(len(dictionary["10"]))+float(len(dictionary["11"])))/grandtotal
        if (last):
                ## checks if this is the dependat variable because specific entropy for this is slightly different than dependant var
                ## since no calculating for 10, 11,00, 01, just 1 or 0 probability
            spec_cond_entropy_total=check_for_zero_entropy(probability0,probability1)
            return(spec_cond_entropy_total)
        else:
        ## the formulat for calculating entropy is below in cond)entropy
            spec_cond_entropy0=check_for_zero_entropy(probability00,probability01)
            spec_cond_entropy1=check_for_zero_entropy(probability10,probability11)
            cond_ent=spec_cond_entropy0*(probability0/grandtotal)+spec_cond_entropy1*(probability1/grandtotal)
            return(cond_ent)
            
        return None


def split(df):
    ## create list of column names to use for temporary checking of gain and stats
    list_col=list(df.columns.values)
    ## create cache for the highest value gain as this will be the split
    highest_col={"name":None,"00":[],"01":[],"10":[],"11":[],"gain":0}
    entropy_last=0
    ## loop through each column and attain the possible combinations then send
    ## to calculate information gain
    for i in range(len(list_col)):
        i=len(list_col)-1-i
        ## Note since we are only allowing binary decisions, there are 4 permutations of classifier and dependant variable data
        test_col={"name":df.columns.values[i],"00":[],"01":[],"10":[],"11":[],"gain":0}
        for j in range (len(df.index)):
            if(df.iloc[j,i]==0 and df.iloc[j,len(list_col)-1]==0):
                test_col["00"].append(j)
            elif(df.iloc[j,i]==0 and df.iloc[j,len(list_col)-1]==1):
                test_col["01"].append(j)
            elif(df.iloc[j,i]==1 and df.iloc[j,len(list_col)-1]==0):
                test_col["10"].append(j)
            elif(df.iloc[j,i]==1 and df.iloc[j,len(list_col)-1]==1):
                test_col["11"].append(j)
        if i==len(list_col)-1:
            entropy_last=calc_entropy(test_col,True)
            print("last",entropy_last)
        else:
                ## the total gain is the last entropy minus each classifier's entropy the highest gain is the one we split on
            conditional_ent=calc_entropy(test_col,False)
            gain=entropy_last-conditional_ent
            print("arnold",gain)
            test_col["gain"]=gain
            if (highest_col["gain"]<test_col["gain"]):
                highest_col=copy.deepcopy(test_col)
        print(test_col)
        print(highest_col)
    return(highest_col)
       

 
## Random Forest Algorithm
## THIS IS NOT OUR CODE COPIED FROM https://machinelearningmastery.com/implement-random-forest-scratch-python/ but modified slightly
def random_forest(train, sample_size, n_trees, class_label):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		print(sample)
		treez = tree(sample,class_label)
		trees.append(treez)
	return(trees) 
# Create a random subsample from the dataset with replacement
# Bagging is random sampling both the classifiers and the data used for training
##We did NOT do the random sampling of classifiers ONLY the different data sets
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

 ## this method turns picture to list of 0 or 1 pixels using numpy library              
def pic_to_list(in_list,word, num_pics,one_or_z):
    for i in range(num_pics):
        temp_list=[]
        temp=np.asarray(Image.open(word+' '+ '('+str(i)+')'+'.PNG').convert("P",colors=2))
        ## turning 2d list to 1d list of pixels to stay consistent with this algorithm as it is optimized for oned lists 
        for j in temp:
            for t in j:
                temp_list.append(t)
        temp_list.append(one_or_z)
        in_list.append(temp_list)
    return(in_list)       
    
## in this main method we use it to put the pictures we want and train the machine learning model
def main():
    traina=[]
    print("Growing forest ")
    traina=pic_to_list(traina,'pica',21,1)
    traina=pic_to_list(traina,'picb',21,0)
    traina=pic_to_list(traina,'picc',15,0)
    traina=pic_to_list(traina,'picd',16,0)
    class_labels=[]
    for i in range(len(traina[0])):
        class_labels.append(i)
    
    trainb=[]
    trainb=pic_to_list(trainb,'pica',21,0)
    trainb=pic_to_list(trainb,'picb',21,1)
    trainc=pic_to_list(trainb,'picc',15,0)
    trainb=pic_to_list(trainb,'picd',16,0)
    
    trainc=[]
    trainc=pic_to_list(trainc,'pica',21,0)
    trainc=pic_to_list(trainc,'picb',21,0)
    trainc=pic_to_list(trainc,'picc',15,1)
    trainc=pic_to_list(trainc,'picd',16,0)

    
    traind=[]
    traind=pic_to_list(traind,'pica',21,0)
    traind=pic_to_list(traind,'picb',21,0)
    traind=pic_to_list(traind,'picc',15,0)
    traind=pic_to_list(traind,'picd',16,1)
    
    foresta=random_forest(traina[:-2],.7,8,class_labels)
    forestb=random_forest(trainb[:-2],.7,8,class_labels)
    forestc=random_forest(trainc[:-2],.7,8,class_labels)
    forestd=random_forest(traind[:-2],.7,8,class_labels)
    print("these are your forest")
    print(foresta)
    print(forestb)
    print(forestc)
    print(forestd)
    ## these are the forest models for each letter inputed into data models. In this model we then send it to pickle file so the prediction file can use it without
    ## having to wait for this file to run
    datamodels = [foresta,forestb,forestc,forestd]

    pickle_out = open("dict1.pickle","wb")
    pickle.dump(datamodels, pickle_out)
    pickle_out.close()

   

main()
