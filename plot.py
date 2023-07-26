import pickle
import numpy as np
import matplotlib.pyplot as plt

#tna=np.array ([[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]])

#for team_idx in range (0,5):
    #for z in range (0,4):
        #agent_idx=tna[team_idx,z]

team_idx= 1

agent_idx1=0
agent_idx2=1
agent_idx3=3
agent_idx4=4

file=open('SSAnalysis' + str(agent_idx1) + str(team_idx), 'rb')   
data1=pickle.load(file)
file.close()

file=open('SSAnalysis' + str(agent_idx2) + str(team_idx), 'rb')
data2=pickle.load(file)        
file.close()

file=open('SSAnalysis' + str(agent_idx3) + str(team_idx), 'rb')
data3=pickle.load(file)
file.close()

file=open('SSAnalysis' + str(agent_idx4) + str(team_idx), 'rb')
data4=pickle.load(file)
file.close()

gen=np.zeros(81)


for i in range(0,81,1):
        gen[i]=i*50

x = gen

plt.plot(x, data1, color='green', label = 'Agent' + ' ' + str(agent_idx1))
plt.plot(x, data2, color='red', label = 'Agent' + ' ' + str(agent_idx2))
plt.plot(x, data3, color='blue', label = 'Agent' + ' ' + str(agent_idx3))
plt.plot(x, data4, color='tab:orange', label = 'Agent' + ' ' + str(agent_idx4))

plt.ylim([0,100])
plt.xlim([0,4000])        
plt.legend(loc='upper left')
plt.xlabel("Generation")
plt.ylabel("Percent Alignment")
plt.title("Percent Alignment for Team " + str(team_idx))
plt.show()