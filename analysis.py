from alignment import load_data,eval
import numpy as np
import pickle

sample_size = 10000 #Cannot be lower then 100, or else randint fuction freezes
Reward_Data= np.zeros((sample_size,2))
Percent_Aligned_arr= np.zeros((80,1))
Alignment_Data= np.zeros((sample_size,3))
Calc= np.zeros((sample_size,2))
count=0
row_index=0
index=0

file=open('SSAnalysis', 'wb')

for j in range (0,4001,50): 

    if __name__=="__main__":
        agent_idx=3 #which agent
        team_idx=2 #which team
        env,pos,teams,net=load_data(n_agents=5,agent_idx=agent_idx,n_actors=4,iteration=0, generation=j)
        
        for row_index in range (sample_size):
            x=np.random.uniform(-5,35) # -5 to 35 ish
            y=np.random.uniform(-5,35) # -5 to 35 ish
            t=np.random.uniform(-np.pi,np.pi) #-pi to pi
            state,G = eval(x,y,t,team_idx,agent_idx,env,pos,teams)
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

        print('%', Percent_Aligned)
    
    i = (j//50) - 1
    Percent_Aligned_arr [i]= Percent_Aligned

print (Percent_Aligned_arr)

#Write data into pickle file
#pickle.dump(Percent_Aligned_arr[:],file)

file.close()
        