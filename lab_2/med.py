
# coding: utf-8

# In[7]:



def medistance(source, target):
    #length of the source and target assigned to n and m respecitvely
    n= len(source) 
    m= len(target)
    
    #initializing the costs of insertion, substitution and deletion
    ins_cost=1
    sub_cost=2
    del_cost=1
    
    #creation of the distance matrix using a 2-D array
    D=[[0 for a in range(n+1)] for a in range(m+1)]
    
    #initializing the zeroth row 
    for i in range (0, n+1):
        D[i][0]=i
        
    #initializing the zeroth column
    for j in range (0, m+1):
        D[0][j]=j
        
    #Recurrence relation
    for i in range (1, n+1):
        for j in range (1, m+1):
            if source[i-1]==target[j-1]:
                D[i][j]=D[i-1][j-1]
            else:
                D[i][j]=min(D[i-1][j]+del_cost,
                                D[i-1][j-1]+sub_cost,
                                    D[i][j-1]+ins_cost)
    
    #Termination
    return D [n][m]


s="intention"
t="execution"

print ("Minimum edit distance between", s, "and", t, "is", medistance(s,t))

