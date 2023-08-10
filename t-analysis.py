import numpy as np
import torch
import pickle
from os import killpg
import os
import matplotlib.pyplot as plt

def getarr (agent_idx,g):
    filepath = r"T/"+"Global" + str(agent_idx)+"-"+str(g)+'.pkl'
    file=open(filepath, 'rb')
    bigG=pickle.load(file)
    file.close
                    
    filepath = r"T/"+"Local"+str(agent_idx)+"-"+str(g)+'.pkl'
    file=open(filepath, 'rb')
    lilg=pickle.load(file)
    file.close

    return lilg,bigG

#----------------

agent_idx=0
numtraj=(4*50)
lilg=np.zeros((30,numtraj)) #Local Rewards of all trajectories
bigG=np.zeros((30,numtraj)) #Global Rewards for all trajectories
Alignment_Data= np.zeros(numtraj) #To store the alignment calculations
Calc= np.zeros(numtraj)
GlobRwrds=np.zeros(numtraj) 
Percent_Aligned_arr= np.zeros((81,2))
ErrArr=np.zeros(100)
index=0
count=0
gen=np.zeros((81))

for i in range(0,81):
 gen[i]=i*50

for g in range (0,4001,50):
    lilg,bigG=getarr(agent_idx,g)
    LocRwrds=np.max(lilg, axis=0)    
    GCoods=np.argmax(lilg, axis=0)

    for i in range (0,(4*50)):
        GlobRwrds[i]=bigG[GCoods[i],i]

    for x in range (0,100):
        while index < numtraj:
            idx1=np.random.randint(0,(numtraj-1))
            idx2=np.random.randint(0,(numtraj-1))
            GR1 = GlobRwrds[idx1]
            GR2 = GlobRwrds[idx2]
                            
            while GR1==GR2:
                idx2=np.random.randint(0,(numtraj-1))
                GR2 = GlobRwrds[idx2]
                if count==(numtraj-1):
                    break
                count=count+1
                                
            if count != numtraj:
                LR1 = LocRwrds[idx1]
                LR2 = LocRwrds[idx2]

                aligned= (LR1 - LR2)*(GR1 - GR2)
                if aligned in Calc:
                    break
                if aligned > 0:
                    Alignment_Data[index]=1
                elif aligned < 0:
                    Alignment_Data[index]=0

                Calc[index]=aligned

                index=index+1

        #Sum the aligned data and calculate percent aligned
        Sum_Aligned= Alignment_Data[:].sum() 
        ErrArr[x]= (Sum_Aligned/(index+1)) *100

        #reset arrays
        Alignment_Data= np.zeros(numtraj) #To store the alignment calculations
        Calc= np.zeros(numtraj)
        count=0
        index=0

    h=g//50
    Percent_Aligned_arr[h,0]=np.mean(ErrArr)

y= Percent_Aligned_arr[:,0]

plt.plot(gen, y, color='green', label = 'Agent' + ' ' + str(agent_idx))

plt.ylim([0,100])
plt.xlim([0,4000])        
plt.legend(loc='upper left')
plt.xlabel("Generation")
plt.ylabel("Percent Alignment")
plt.title("Percent Alignment for Agent " + str(agent_idx))


plt.show()
