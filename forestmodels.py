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


## the function to turn a image from a pixel image to a 1Dimensional list of 1s and 0s
def pic_to_list(in_list,word,index):
    ##initializing a list and using np library to put image into 2d array
    temp_list=[]
    try:
        temp=np.asarray(Image.open(word+' '+ '('+str(index)+')'+'.PNG').convert("P",colors=2))
        ## For ease of access I want in only 1D list so loop through put into list
        for j in temp:
            for t in j:
                temp_list.append(t)
        temp_list.append(index)
        ## append this list to the list you pass in nd return
        in_list.append(temp_list)
        return(in_list)
    except:
        return ("Please read the comment in the main")
## from a of letter predictors it determines the inner list with the most 1's, this index passed to list of letters
## is the letter of prediction
def which_letter(list_of_letters,list_of_pred):
    ## large_count is to see if two list have the sum if so two predicted letters means our machine failed see 
    largest=sum(list_of_pred[0])
    largestindex=0
    large_count=0
    for i in range(len(list_of_letters)-1):
        i=i+1
        current_total=sum(list_of_pred[i])
        if current_total>largest:
            largest=current_total
            largestindex=i
            large_count=0
        elif current_total==largest:
            largestindex=largestindex+1
    ##stated above
    if large_count>0:
        return "OUR MACHINE HAS FAILED TO PREDICT PROPERLY"
    else:
        return list_of_letters[largestindex]
            
    
            
## Tree Traversal in log(n) time called from
def predict(node,to_predict):
    try:
        if to_predict[node.getData()].values[0]==0:
            if node.getLeft()!= None:
                left= predict(node.getLeft(),to_predict)
                return left
            
            else:
                return node.getData()
        if to_predict[node.getData()].values[0]==1:
            if node.getRight()!= None:
                right=predict(node.getRight(),to_predict)
                return right
            else:
                return node.getData()
    except:
        return node.getData()
            

## forest list is a list of random forest models for each letter ,i.e. a or not a, b or not b, etc.
##containing as many nodes as their are trees in each forest. What this method does is determine
## the results of running the picture through the list of nodes and in each takes the list which declares that
## the most that the picture is the letter and returns that letter along with the predictions per list of trees for that letter
## Pic name and number need to be passed in a certain way such that the picture in the folder is named
## picname (number).PNG as you will see in predict file, list of letters is the letter which corresponds to
## the index of the predictor in the forest list. i.e if forest_list [0] determines a or not a list_of_letters must be
## list_of_letters[0] =a
def enter_forest(forest_list,pic_name,number,list_of_letters):
    final_result=[]
    class_labels=[]
    picture_pixels=[]
    ## send pixel image to list of ones and zeros
    picture_pixels=pic_to_list(picture_pixels,pic_name,number)
    for i in range(len(picture_pixels[0])):
        class_labels.append(i)
    ## using a dataframe for ease of access of x and y categories
    dfa=pd.DataFrame(data=picture_pixels, columns=class_labels)
    for i in forest_list:
        ## current letter is the current list of trees that the program is on remember forest list is a list of
        ## letter predictors
        current_letter=[]
        for j in i:
            ## runs through each list of trees for that specific letter and appends wheter is letter or not letter
            current_letter.append(predict(j,dfa))
        final_result.append(current_letter)
    ## looks at sum of each list in the final result and sums them up- the max decides which is the index
        ## passed into the corresponding list_of_letters which passes out a letter to be returned
    letter_prediction=which_letter(list_of_letters,final_result)
    return(final_result,letter_prediction)
    
    
     
def main():
    if __name__=='__main__':
        with open('dict.pickle', 'rb') as f:
            pickle_list = pickle.load(f)
    ##pickle list is the list of trained random forest models with their trees for each letter in this model there is the letter a,b,c,d
    ## passed into predcition along with the photoname, letter, and corresponding letters of this list, the enterforest method is called
   
    print("\n in order of [a,b,c,d] the prediction count for your letter variancea by each letter forest and their trees, followed by prediction result is")
    ##Note your photo must be in the folder in a format of 'name(#).PNG' without the quotes that I have here or else it wont work
    predictions,letter=enter_forest(pickle_list,"img/dirty",0,['a','b','c','d'])
    print(predictions ,"\n",letter)
      
   
        
    
   


main()
