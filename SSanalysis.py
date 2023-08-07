from alignment import load_data,eval
import numpy as np
import pickle
from os import killpg
import os

def save (team_idx, agent_idx,Percent_Aligned_arr):
    folder="SS/"
    if not os.path.exists("SS"):
        os.makedirs("SS")
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = r"SS/"+str(team_idx)+"-"+str(agent_idx)+'.pkl'
    file=open(filepath, 'wb')
    pickle.dump(Percent_Aligned_arr,file)
    file.close

#-----------------------------

sample_size = 10000 #Cannot be lower then 100, or else randint fuction freezes
Reward_Data= np.zeros((sample_size,2))
Percent_Aligned_arr= np.zeros((4001,1))
Alignment_Data= np.zeros((sample_size,3))
Calc= np.zeros((sample_size,2))
count=0
index=0
tna=np.array ([[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]])

team_idx= 0

for team_idx in range (0,5):
#Cycle through team indexes
    for cidx in range (0,4):
        agent_idx= tna[team_idx,cidx]
        print(agent_idx,team_idx)
        for j in range (0,4001,50):
            if __name__=="__main__":
                env,pos,teams,net=load_data(n_agents=5,agent_idx=agent_idx,n_actors=4,iteration=0, generation=j)
            
            for g in range (0,4001):
                for row_index in range (0,sample_size):
                    x=np.random.uniform(-5,35) # -5 to 35 ish
                    y=np.random.uniform(-5,35) # -5 to 35 ish
                    t=np.random.uniform(-np.pi,np.pi) #-pi to pi
                    state,G = eval(x,y,t,team_idx,agent_idx,env,pos,teams,generation=g)
                    G_estimate=net.feed(state)[0,0]
                    Reward_Data[row_index,0] = G
                    Reward_Data[row_index,1] = G_estimate
                        
                while index < sample_size:
                    test_index=np.random.randint(0,(sample_size-1))
                    row_index=np.random.randint(0,(sample_size-1))
                    G1 = Reward_Data[row_index,0]
                    G2 = Reward_Data[test_index,0]
                            
                    while G1==G2:
                        test_index=np.random.randint(0,sample_size-1)
                        G2 = Reward_Data[test_index,0]
                        if count==(sample_size-1):
                            break
                        count=count+1
                                
                    if count != sample_size:
                        GE1 = Reward_Data[row_index,1]
                        GE2 = Reward_Data[test_index,1]

                        aligned= (GE1 - GE2)*(G1 - G2)
                        if aligned in Calc:
                            break
                                    
                        if aligned > 0:
                            Alignment_Data[index,0]=1
                        elif aligned < 0:
                            Alignment_Data[index,0]=0

                        Calc[index]=aligned
                        Alignment_Data[index,1]=row_index
                        Alignment_Data[index,2]=test_index

                        index=index+1
            #Sum the aligned data and calculate percent aligned
            Sum_Aligned= Alignment_Data[:,0].sum() 
            Percent_Aligned= (Sum_Aligned/(index+1)) *100

            print('%', Percent_Aligned,j)
            
            Percent_Aligned_arr [g]= Percent_Aligned

            #Write data into pickle file
            save(team_idx,agent_idx,Percent_Aligned_arr)

            #reset arrays
            Reward_Data= np.zeros((sample_size,2))
            Percent_Aligned_arr= np.zeros((4001,1))
            Alignment_Data= np.zeros((sample_size,3))
            Calc= np.zeros((sample_size,2))
            count=0
            index=0
                