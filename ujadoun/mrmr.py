import json
import numpy as np
import math

#Generate a list of the Featuress in m (rows) using mrmr algorithm.
#Return the list of the attribute for the generated Featuress
def select(m):
    X=m[1:]
    Y=m[0]
    mrmrFeatures=[]
    for i in range(len(X)):
        maxEntropy=[]
        attribute=[]
        entropyC=entropy(Y)
        for j in range(0,len(X)):
            if j+1 not in mrmrFeatures:
                if mrmrFeatures !=[]:
                    entropyC_G=mutualInfo(Y,X[j])
                    gainYJ=entropyC-entropyC_G
                    gainOth=0
                    for feature in mrmrFeatures:
                        entropyV=entropy(X[feature-1])
                        entropyVJ=mutualInfo(X[feature-1],X[j])
                        gainOth+=entropyV-entropyVJ
                    maxEntropy.append(len(mrmrFeatures)*gainYJ/gainOth)
                    attribute.append(j)
                else:
                    entropyC_G=mutualInfo(Y,X[j])
                    maxEntropy.append(entropyC-entropyC_G)
                    attribute.append(j)


        maxFeatureEntropy=max(maxEntropy)
        maxFeatureEntropyIndex=maxEntropy.index(maxFeatureEntropy)
        mrmrFeatures.append(attribute[maxFeatureEntropyIndex]+1)
    return mrmrFeatures



#calculate mutual information for r1 and r2. 
def mutualInfo(r1, r2):
    entropy_y=entropy(r1)
    entropy_yg=0
    den_S=len(r1)

    for i in np.unique(r1):
        for j in np.unique(r2):
            SAB=r1[r2==j][r1[r2==j]==i]
            SB=r2[r2==j]
            numSAB=len(SAB)
            denSB=len(SB)
            entropy_yg+=(-(numSAB)/den_S)*np.log2((numSAB)/(denSB))

    return entropy_yg


#Calculate the entropy for a, which is a set of values for a single Features.
def entropy(a):
    entropyC=0
    uniqueElement=set()
    for i in np.unique(a):
        uniqueElement.add(len(a[a==i]))
    
    for unique in uniqueElement:
        entropyC+=((-unique/len(a))*math.log2(unique/len(a)))


    return entropyC



