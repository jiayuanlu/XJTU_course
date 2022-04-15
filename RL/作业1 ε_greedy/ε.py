import numpy as np
import matplotlib.pyplot as plt

action=[1,2,3,4,5,6,7,8,9,10]
reward={}
prob={}
reward={1:10,2:1,3:4,4:6,5:8,6:7,7:3,8:2,9:9,10:5}
prob={1:0.1,2:0.8,3:0.6,4:0.5,5:0.3,6:0.35,7:0.7,8:0.75,9:0.15,10:0.55}
# epsilon=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
epsilon=[0,0.01,0.1]
times=10000
num={}
x=[]
y=[]
x=[[0]*len(epsilon) for i in range(int(times/10))]
y=[[0]*len(epsilon) for i in range(int(times/10))]
Q={}
for e in range(len(epsilon)):
    R=0
    avg_R=0
    for i in reward.keys():
        Q[i]=0
    for i in reward.keys():
        num[i]=0
    for i in range(times):
        if np.random.random()<epsilon[e]:
            A=np.random.choice(action)
        else:
            A=max(Q,key=Q.get)
        R=R+(reward[A]-R)/(i+1)
        num[A]+=1
        Q[A]+=(R-Q[A])/num[A]
        if (i%10)==0:
            x[e].append(i)
            y[e].append(R)
    plt.plot(x[e],y[e])
plt.xlabel("Steps")
plt.ylabel("Average Reward")
# plt.legend(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
plt.legend(['0','0.01','0.1'])
plt.show()
