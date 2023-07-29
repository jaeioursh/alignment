from mtl import make_env

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pickle

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

def getcood(team_idx,agent_idx,env,position,teams,generation,time=-1):
    pos,rot=position[generation][team_idx][time]
    [sin,cos]=rot[agent_idx,:]
    [x,y]=pos[agent_idx,:]

    return x,y,sin,cos

#----------CODE------------#

rewards1=np.zeros((30,2))
rewards2=np.zeros((30,2))

if __name__=="__main__":
    agent_idx=0
    team_idx1=0
    team_idx2=1
    generation=0
    env,pos,teams,net=load_data(n_agents=5,agent_idx=agent_idx,n_actors=4,iteration=0,generation=3000)
    
    for i in range (0,30):
        x,y,sin,cos= getcood(team_idx1,agent_idx,env,pos,teams,generation=3000,time=i)
        state1,G1= evaltraj(team_idx2,agent_idx,env,pos,teams,generation=3000,time=i)
        state2,G2= evaltrajc(x,y,sin,cos,team_idx2,agent_idx,env,pos,teams,generation=3000,time=i)
        g1=net.feed(state1)[0,0]
        g2=net.feed(state2)[0,0]
        rewards1[i,0]=G1
        rewards1[i,1]=g1
        rewards2[i,0]=G2
        rewards2[i,1]=g2

print (rewards1)
print (rewards2)

    