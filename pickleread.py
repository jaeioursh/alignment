import numpy as np
import pickle

file=open('SSAnalysis', 'rb')

data=pickle.load(file)
    
file.close()

file=open('pickleread.txt','w+')

i=0
for i in range(10000):
    content=str(data[i,:])
    file.write(content)
    file.write('\n')
    i+=1

file.close()