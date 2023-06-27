from alignment import load_data,eval
import numpy as np

sample_size = 100
Reward_Data= np.zeros((sample_size,2))
State_Array= np.zeros((sample_size,3))
Alignment_Data= np.zeros((sample_size,3))
row_index=0
index=0

if __name__=="__main__":
    agent_idx=0
    team_idx=0
    env,pos,teams,net=load_data(n_agents=5,agent_idx=agent_idx,n_actors=4,iteration=0,generation=1000)
    
    for row_index in range (sample_size):
        x=np.random.uniform(-5,35) # -5 to 35 ish
        y=np.random.uniform(-5,35) # -5 to 35 ish
        t=np.random.uniform(-np.pi,np.pi) #-pi to pi
        state,G = eval(x,y,t,team_idx,agent_idx,env,pos,teams)
        G_estimate=net.feed(state)[0,0]
        State_Array[row_index,0]=x
        State_Array[row_index,1]=y
        State_Array[row_index,2]=t
        Reward_Data[row_index,0] = G
        Reward_Data[row_index,1] = G_estimate
        
    Reward_Data[9,0]=2
    
    while index < sample_size:
        test_index=np.random.randint(0,sample_size-1)
        row_index=np.random.randint(0,sample_size-1)
        G1 = Reward_Data[row_index,0]
        G2 = Reward_Data[test_index,0]
        
        while G1==G2:
            test_index=np.random.randint(0,sample_size-1)
            G2 = Reward_Data[test_index,0]

        GE1 = Reward_Data[row_index,1]
        GE2 = Reward_Data[test_index,1]

        aligned= (GE1 - GE2)*(G1 - G2)
        if aligned > 0:
            Alignment_Data[index,0]=1
        elif aligned < 0:
            Alignment_Data[index,0]=0

        Alignment_Data[index,1]=row_index
        Alignment_Data[index,2]=test_index

        index=index+1
    
    #Sum the aligned data and calculate percent aligned
    Sum_Aligned= Alignment_Data[:,0].sum() 
    Percent_Aligned= (Sum_Aligned/sample_size) *100

    print('%', Percent_Aligned)

    #Save Data to File
    file = open("Rewards.txt", "w+")
    content = str(Reward_Data)
    file.write(content)
    file.close()

    file = open("Alignment-Calc.txt", "w+")
    content = str(Alignment_Data)
    file.write(content)
    file.close()

    file = open("Position.txt", "w+")
    content = str(State_Array)
    file.write(content)
    file.close()