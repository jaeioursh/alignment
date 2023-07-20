import pickle
import numpy as np
import matplotlib.pyplot as plt


file=open('SSAnalysis', 'rb')

data=pickle.load(file)

pa= np.zeros((1,81))
gen=np.zeros((1,81))

for i in range(0,81,1):
    pa[0,i]=data[i]
    gen[0,i]=i*50

file.close()


N = 50
x = gen [0,:]
y = pa [0,:]

plt.plot(x, y, color='tab:green')
plt.ylim([0,100])
plt.xlim([0,4000])
plt.xlabel("Generation")
plt.ylabel("Percent Alignment")
plt.title("percent alignment for agent 0 in team 2")
plt.show()