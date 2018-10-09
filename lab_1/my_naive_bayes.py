
# coding: utf-8

# In[1]:


import sys
import os
import re
import math
import string
from collections import Counter

classes=['0', '1']
logprior=[]
file_to_read= 'amazon_cells_labelled.txt'
positives=[] #list for positive sentences
negatives=[] #list for negative sentences

positive_dict={} #positive dictionary that contains a word as key and its loglikelihood as the value
negative_dict={} #negative dictionary that contains a word as key and its loglikelihood as the value

#function to calculate frequency of all the words in a specified list of classes positive or negative
def freq(list):
    wordfreq = []
    for x in list:
        wordlist = x.split()
        for w in wordlist:
            wordfreq.append(wordlist.count(w))
            
    b=sum(wordfreq) #sum of all the frequencies
    return (b)


# In[2]:


#function that returns a bag of words and its length
def extract():
    with open (file_to_read) as file:
        text=file.read()
        words = sorted(set(re.split(r'\W+', text))) #sorts, put into set, split into words and gets rid of punctuations and duplicates
        words=[word.lower()for word in words]
        vocab_length=len(words)
        return (words,vocab_length)
    file.close()


# In[3]:


#function that uniquely identifies words and places them in a vocabulary list
def classifier():
    bag_of_words, bag_length=extract() #gets the bag of words and the length of the bag of words from the extract() function
    
    positive_counter=0 #keep track of number of positive sentences
    negative_counter=0 #keep track of number of negative sentences
    
    with open (file_to_read, 'r+') as file:
        for line in file:
            #classifies senteneces to positive and adds to positive list
            if (line.find('1'))!=-1:
                positive_counter +=1
                positives.append(line)
                
            #classifies senteneces to negative and adds to negative list
            elif (line.find('0'))!=-1:
                negative_counter+=1
                negatives.append(line)
        
        n_doc=positive_counter+negative_counter #total number of sentences in the file
        for x in classes:
            if x is '0':
                logprior_negative=math.log(negative_counter/n_doc) #finding the logprior of the negative class
                logprior.append(logprior_negative) #appends the result into a log prior list
            elif x is '1':
                logprior_positive=math.log(positive_counter/n_doc) #finding the logprior of the positive class
                logprior.append(logprior_positive)
                
        frequency_total_positive=freq(positives) #frequency of all the words in positive
        frequency_total_negative=freq(negatives) #frequency of all the words in negative
                
        #counting the number of occurences of a word in the two classes and place the word and its loglikelihood in a dictionary
        for x in bag_of_words:
            x_count_pos=0 #sum of total occurences of the words in positive class
            x_count_neg=0 #sum of total occurneces of the words in negative class
            
            for line in positives: 
                line=line.split(" ") #lines are put into lists for easy management
                line_count=line.count(x)
                x_count_pos += line_count
            for line in negatives:
                line=line.split(" ")
                line_count=line.count(x)
                x_count_neg += line_count
            
            likelihood_pos=math.log((x_count_pos + 1)/(frequency_total_positive + bag_length)) #calculate loglikelihood of a word in the positive class
            positive_dict[x]=likelihood_pos #put the word and its loglikelihood as the key and value respecitively
            
            likelihood_neg=math.log((x_count_neg + 1)/(frequency_total_negative + bag_length)) #calculate loglikelihood of a word in the negative class
            negative_dict[x]=likelihood_neg #put the word and its loglikelihood as the key and value respecitively
            
        
        #return (positive_counter, negative_counter, frequency_total_positive, frequency_total_negative, bag_length)
        return (logprior, positive_dict, negative_dict)
    file.close()


# In[4]:


def test_naive(file):
    bag_of_words, bag_length=extract()
    predict_array=[]
    logprior, positive_dict, negative_dict=classifier()
    file = open(file, "r")
    positive_prob=logprior[1]
    negative_prob=logprior[0]
    for line in file:
        refined_line=line.lower()
        refined_line=refined_line.translate(str.maketrans("","",string.punctuation))
        refined_line=refined_line.split(" ")
        #print (refined_line)
            
        for i in refined_line:
            if i in bag_of_words:
                positive_prob += positive_dict[i]
                negative_prob += negative_dict[i]
                    
        if positive_prob > negative_prob:
            predict_array.append('1')
        else:
            predict_array.append('0')
    
    #print(predict_array)
    
    f=open("results.txt", "w")
    
    for item in predict_array:
        f.write(item+"\n")
    f.close()


# In[5]:


def main():
    #classifier()
    #print (logprior)
    logprior, positive_dict, negative_dict= classifier()
    print (logprior)
    
    #test_naive('yelp_labelled.txt')
    test_naive(sys.argv[1])
    
    '''positive_class, negative_class, positive_freq, negative_freq, bag_length=classifier() #works
    print('Positive Class length is', positive_class) #works
    print('Negative Class length is', negative_class)#works
    print('Total Number of positive class words frequency is', positive_freq) #worrks
    print('Total Number of negative class words frequency is', negative_freq) #works
    print('Bag of words length is', bag_length) #works'''
    

main()

