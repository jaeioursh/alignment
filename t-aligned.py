from mtl import make_env

import numpy as np
import torch
import pickle
from os import killpg
import os

import code.reward_2 as reward
from teaming import logger
from teaming.learnmtl import Net


def load_data(n_agents=5,agent_idx=0,n_actors=4,iteration=0,generation=0,q=3,hidden=20):
    fname="tests/"+str(n_agents)+"-"+str(n_actors)+"-"+str(iteration)+"-"+str(q)

    log = logger.logger()
    log.load(fname+"/data.pkl")
    env=make_env(n_actors)
    pos=log.pull("position")
    teams=np.array(log.pull("types")[0])

    net=Net(hidden)
    net.model.load_state_dict(torch.load(fname+"/"+str(generation)+".mdl")[agent_idx])
    return env,pos,teams,net

def evaltrajc(x,y,sin,cos,team_idx,agent_idx,env,position,teams,generation,time=-1):
    pos,rot=position[generation][team_idx][time]
    pos,rot=pos.copy(),rot.copy()
    pos[teams[team_idx]==agent_idx,:]=[x,y]
    env.data["Agent Orientations"]=rot
    rot[teams[team_idx]==agent_idx,:]=[sin,cos]
    env.data["Agent Positions"]=pos
    env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
    env.data["Steps"]=0
    env.data["Observation Function"](env.data)
    z=env.data["Agent Observations"]
    s=z[teams[team_idx]==agent_idx]

    env.data["Reward Function"](env.data)
    g=env.data["Global Reward"]

    return s,g

def evaltraj(team_idx,agent_idx,env,position,teams,generation,time=-1):
    pos,rot=position[generation][team_idx][time]
    pos,rot=pos.copy(),rot.copy()
    env.data["Agent Orientations"]=rot
    env.data["Agent Positions"]=pos
    env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
    env.data["Steps"]=0
    env.data["Observation Function"](env.data)
    z=env.data["Agent Observations"]
    s=z[teams[team_idx]==agent_idx]

    env.data["Reward Function"](env.data)
    g=env.data["Global Reward"]

    return s,g

def getcood(team_idx,agent_idx,AgentPositionInTeam,position,teams,generation,time=-1):
    pos,rot=position[generation][team_idx][time]
    j=AgentPositionInTeam[agent_idx,team_idx1]
    [sin,cos]=rot[j,:]
    [x,y]=pos[j,:]

    return x,y,sin,cos

def save (agent_idx,g,bigG,lilg):
    folder="T/"
    if not os.path.exists("T"):
        os.makedirs("T")
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = r"T/"+"Global" + str(agent_idx)+"-"+str(g)+'.pkl'
    file=open(filepath, 'wb')
    pickle.dump(bigG,file)
    file.close
                    
    filepath = r"T/"+"Local"+str(agent_idx)+"-"+str(g)+'.pkl'
    file=open(filepath, 'wb')
    pickle.dump(lilg,file)
    file.close


#----------CODE------------#
numtraj=(4*50)
lilg=np.zeros((30,numtraj)) #Local Rewards of all trajectories
bigG=np.zeros((30,numtraj)) #Global Rewards for all trajectories
GlobRwrds=np.zeros(numtraj) #Max Global Rewards
Alignment_Data= np.zeros(numtraj) #To store the alignment calculations
Calc= np.zeros(numtraj) #To store calculations to check for repeats
#Percent_Aligned_arr= np.zeros((81,1))
index=0
count=0

ant=np.array ([[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]])
AgentPositionInTeam=np.array([[0,0,0,0,9],[1,1,1,9,0],[2,2,9,1,1],[3,9,2,2,2],[9,3,3,3,3]])

#for agent_idx in range (0,5):
agent_idx=0
team_idx1= ant[agent_idx,0]
team_idx2= ant[agent_idx,1]
team_idx3= ant[agent_idx,2]
team_idx4= ant[agent_idx,3]



if __name__=="__main__":
            for g in range (0,4001):
                print (g)
                if g%50==0:
                    env,pos,teams,net=load_data(n_agents=5,agent_idx=agent_idx,n_actors=4,iteration=0,generation=g)
                    j=g
                    save(agent_idx,g,bigG,lilg)


                for i in range (0,30):
                    #Get Control Data
                    state1,G1= evaltraj(team_idx1,agent_idx,env,pos,teams,generation=g,time=i)
                    g1=net.feed(state1)[0,0]
                    idx1=0+((g-j)*4)
                    bigG[i,idx1]=G1
                    lilg[i,idx1]=g1

                    #Get Data for Team 2 in Control
                    x,y,sin,cos= getcood(team_idx2,agent_idx,AgentPositionInTeam,pos,teams,generation=g,time=-1)
                    state2,G2= evaltrajc(x,y,sin,cos,team_idx1,agent_idx,env,pos,teams,generation=g,time=i)
                    g2=net.feed(state2)[0,0]
                    idx2=1+((g-j)*4)
                    bigG[i,idx2]=G2
                    lilg[i,idx2]=g2

                    #Get Data for Team 3 in Control
                    x,y,sin,cos= getcood(team_idx3,agent_idx,AgentPositionInTeam,pos,teams,generation=g,time=-1)
                    state3,G3= evaltrajc(x,y,sin,cos,team_idx1,agent_idx,env,pos,teams,generation=g,time=i)
                    g3=net.feed(state3)[0,0]
                    idx3=2+((g-j)*4)
                    bigG[i,idx3]=G3
                    lilg[i,idx3]=g3

                    #Get Data for Team 4 in Control
                    x,y,sin,cos= getcood(team_idx4,agent_idx,AgentPositionInTeam,pos,teams,generation=g,time=-1)
                    state4,G4= evaltrajc(x,y,sin,cos,team_idx1,agent_idx,env,pos,teams,generation=g,time=i)
                    g4=net.feed(state4)[0,0]
                    idx4=3+((g-j)*4)
                    bigG[i,idx4]=G4
                    lilg[i,idx4]=g4
                        

