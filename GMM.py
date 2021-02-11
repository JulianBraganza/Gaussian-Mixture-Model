import numpy as np
import scipy

class GMM:
    def __init__(self,D, K):
        
        self.pis = np.zeros(K)
        self.mus = np.zeros((K,D))
        self.sigmas = np.zeros((K,D,D))
        
    def get_params(self):
        return [self.pis,self.mus,self.sigmas]
        
    def km_e_step(self,X):
        N = X.shape[0]
        K = self.mus.shape[0]
        dists = np.zeros((N,K))
        for k in range(K):
            dists[:,k] = np.linalg.norm(X - self.mus[k,:], axis=1)
            R_new = np.eye(K)[np.argmin(dists, axis = 1).astype(int)]
        return R_new

    def km_m_step(self,X,R):
        N = X.shape[0]
        K = self.mus.shape[0]
        mus_new = np.zeros(self.mus.shape)
        for k in range(K):
            mus_new[k,:] = np.sum(X*np.reshape(R[:,k],(N,1)),axis=0)/np.sum(R[:,k])
        return mus_new
    
    def normal_density(self, x,mu,sigma):
        d = mu.shape[0]
        c = np.pi**(-d/2)*np.linalg.det(sigma)**(-0.5)
        y = np.reshape(x - mu, (x.shape[0],1))
        if d > 100:
            L = np.linalg.cholesky(sigma)
            x = scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,y,lower=True))
            z = np.dot(y.T,x)
        else:
            z = np.dot(y.T,np.dot(np.linalg.inv(sigma),y))
        return c*np.exp(-0.5*z)

    def log_likelihood(self, X):
        def log_sum(x, mus, sigmas, pis):
            val = 0
            K = self.mus.shape[0]
            for k in range(K):
                val += pis[k]*self.normal_density(x,mus[k,:],sigmas[k,:,:])
            return np.log(val)
        return np.sum(np.apply_along_axis(log_sum, 1, X, mus = self.mus, 
                                          sigmas = self.sigmas, pis = self.pis))
    
    def resp(self, x): #Computes responsibilities
        K = self.mus.shape[0]
        gammas = np.zeros(K)
        for k in range(K):
            gammas[k] = self.pis[k]*self.normal_density(x,self.mus[k,:],self.sigmas[k,:,:])        
        return gammas/gammas.sum()

    def em_e_step(self, X):
        return np.apply_along_axis(self.resp, 1, X)

    def em_m_step(self, X,gammas):
        K = self.mus.shape[0]
        N = X.shape[0]
        D = X.shape[1]
        
        mus = np.zeros((K,D))
        sigmas = np.zeros((K,D,D))
        pis = np.zeros(K)
        
        for k in range(K):
            N_k = np.sum(gammas[:,k])
            mus[k,:] = np.sum(X*np.reshape(gammas[:,k],(N,1)),axis = 0)/N_k
            Y = X - np.tile(mus[k,:],(N,1))
            sigmas[k,:,:] = np.dot((Y*np.reshape(gammas[:,k],(N,1))).T,Y)/N_k
            pis[k] = N_k/N
        return mus,sigmas,pis
    
    def train(self,data,tol=10^(-10)):
        X = data
        N = X.shape[0]
        K = self.mus.shape[0]
            
        #Initialize params using the K-means++ algorithm
        self.mus[0,:] = X[np.random.randint(N)]
        for k in range(1,K):
            dists = np.apply_along_axis(np.linalg.norm,1,X-self.mus[k-1,:])
            self.mus[k,:] = X[np.random.choice(np.arange(N),1,p=dists/np.sum(dists))]
            
        for i in range(100):
            gammas = self.km_e_step(X)
            self.mus = self.km_m_step(X,gammas)
        
        assigned = np.argmax(gammas, axis = 1)
        unique, counts = np.unique(assigned, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        for k in range(K):
            self.sigmas[k,:,:] = np.cov(X[assigned == k].T)
            self.pis[k] = counts_dict[k]

        self.pis = self.pis/N
        
        #Begin EM-algoritmn
        converged = False
        niter = 0
        loglik = [0,self.log_likelihood(X)]
        while not converged:
            niter += 1
            loglik[0] = loglik[1]
            
            gammas = self.em_e_step(X)
            self.mus, self.sigmas, self.pis = self.em_m_step(X,gammas)
            
            loglik[1] = self.log_likelihood(X)
            if niter % 100 == 0:
                print("Iteration {}: log-likelihood = {}".format(niter, loglik[1]))
            if abs((loglik[1]-loglik[0])/loglik[1]) < tol:
                converged = True
                print("{} iterations".format(niter))

