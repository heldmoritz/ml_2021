import numpy as np
import pandas as pd
data = pd.read_json("ae_train.json")
type(data)

#This function computes the average of each of the 12 coefficients 
#The input is a datapoint which is an utterance of the vowel /ae/
def average_coefs(utterance):
    #define duration of the utterance
    length = len(utterance)
    
    #define number of coefficients (always 12)
    N = len(utterance[0])
    
    #define the empty output variable
    average_coefficients_per_utterance = np.zeros(N)
    
    #compute for each of the 12 coefficients the average value
    for n in np.arange(N):
        #define the empty coefficients-variable
        coefficients = np.zeros(length)
        
        #define the vector of n-th coefficient with length/dimension = duration of utterance
        for i in np.arange(length):
            coefficients[i] = utterance[i][n]
        
        #compute the average of the n-th coefficient    
        average_coefficients_per_utterance[n] = np.mean(coefficients)   
    
    #define output
    return average_coefficients_per_utterance

#Check the function "average_coefs" with an example
utterance0_of_speaker0 = data.speaker0[0]
try1 = average_coefs(utterance0_of_speaker0)
print("The average coefficients of utterance 0 of speaker 0:")
print(try1)