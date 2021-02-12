import numpy as np
import matplotlib.pyplot as plt
from GMM import GMM

def genData(params,N): #Generates multivariate gaussian clusters
    X = np.random.multivariate_normal(params[0][0],params[0][1],N[0])
    t = np.ones(N[0])*0
    for i in range(1,len(N)):
        X = np.append(X,np.random.multivariate_normal(params[i][0],params[i][1],N[i]),axis=0)
        t = np.append(t,np.ones(N[i])*i)    
    
    return (X,t)

np.random.seed(77)

#Setting cluster parameters
cluster1 = [np.array([0.1,0.1,0.5]),np.array([[10,-7,3],[-7,7,-5],[3,-5,10]])]
cluster2 = [np.array([0.1,0.1,0.5]),np.array([[10,7,3],[7,7,5],[3,5,10]])]
cluster3 = [np.array([0,5,-1]),np.array([[ 8, -0.6, -3. ],[-0.6,  7, 5.1],[-3. , 5.1, 13.4]])]
params = [cluster1,cluster2,cluster3]
N=[200,300,500]

real_data = genData(params,N)

#Plotting the generated data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(real_data[0][:,0], real_data[0][:,1], real_data[0][:,2],marker='o')
ax.text2D(0.05, 0.95, "Original Data", transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

#Showing the real clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(real_data[0][:N[0],0], real_data[0][:N[0],1], real_data[0][:N[0],2],c='r', marker='o')
ax.scatter(real_data[0][N[0]:(N[0]+N[1]),0], real_data[0][N[0]:(N[0]+N[1]),1],
                   real_data[0][N[0]:(N[0]+N[1]),2],c='b', marker='o')
ax.scatter(real_data[0][(N[0]+N[1]):,0], real_data[0][(N[0]+N[1]):,1],
                   real_data[0][(N[0]+N[1]):,2],c='orange', marker='o')
ax.text2D(0.05, 0.95, "Original Clusters", transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

#Fitting model
GMM_model = GMM(dim=3,num_clusters=3)
GMM_model.train(real_data[0],10**(-10))


#Showing the predicted clusters

assignments = (real_data[0], np.argmax(np.apply_along_axis(GMM_model.resp, 1, real_data[0]), axis = 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(assignments[0][np.where(assignments[1]==0),0],
           assignments[0][np.where(assignments[1]==0),1],
           assignments[0][np.where(assignments[1]==0),2],c='green', marker='o')
ax.scatter(assignments[0][np.where(assignments[1]==1),0],
           assignments[0][np.where(assignments[1]==1),1],
           assignments[0][np.where(assignments[1]==1),2],c='purple', marker='o')
ax.scatter(assignments[0][np.where(assignments[1]==2),0],
           assignments[0][np.where(assignments[1]==2),1],
           assignments[0][np.where(assignments[1]==2),2],c='black', marker='o')
ax.text2D(0.05, 0.95, "Predicted Clusters", transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

