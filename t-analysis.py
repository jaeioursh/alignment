import numpy as np
import pickle
from os import killpg
import os

def save (agent_idx,Percent_Aligned_arr):
    folder="Traj/"
    if not os.path.exists("Traj"):
        os.makedirs("Traj")
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = r"Traj/"+str(agent_idx)+'.pkl'
    file=open(filepath, 'wb')
    pickle.dump(Percent_Aligned_arr,file)
    file.close

#----------------

agent_idx=0
numtraj=(4*(4000))+4
lilg=np.zeros((30,numtraj)) #Local Rewards of all trajectories
bigG=np.zeros((30,numtraj)) #Global Rewards for all trajectories
Alignment_Data= np.zeros(numtraj) #To store the alignment calculations
Calc= np.zeros(numtraj)
GlobRwrds=np.zeros(numtraj) 
Percent_Aligned_arr= np.zeros((1,2))
ErrArr=np.zeros(100)
index=0
count=0

file=open('TrajAnalysis1-' + str(agent_idx), 'rb')
bigG=pickle.load(file)      
file.close()

file=open('TrajAnalysis2-' + str(agent_idx), 'rb')
lilg=pickle.load(file)   
file.close()

LocRwrds=np.max(lilg, axis=0)    
GCoods=np.argmax(lilg, axis=0)

for i in range (0,16004):
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
    print(ErrArr[x])

Percent_Aligned_arr[agent_idx,0]=np.mean(ErrArr)
Percent_Aligned_arr[agent_idx,1]=(np.std(ErrArr, ddof=1) / np.sqrt(np.size(ErrArr)))

print (Percent_Aligned_arr)
