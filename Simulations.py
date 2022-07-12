# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:02:10 2022

@author: Dhruv Rathore
"""
import numpy as np
import matplotlib.pyplot as plt

class Simulations:
        
    def __init__(self,N,paths,S_0=1):
        self.N = N
        self.paths = paths
        self.S_0 = S_0
   
    def bm_paths(self):    
        N = self.N; paths = self.paths
        
        global t
        t = np.linspace(0,1,N)
        
        global BM
        BM = np.zeros((N,paths))
        
        
        for i in range(paths):
            inc= np.sqrt(1/N)*np.random.normal(0,1,N-1)
            cumsum=np.cumsum(inc).tolist()
            b=[0] + cumsum
            BM[:,i] = np.array(b)   
            
        plt.plot(t,BM)    
        plt.title("Brownian motion paths simulation")
        plt.xlabel("t")
        plt.ylabel("B(t)")
         
    def get_bm_array(self):
        Simulations.bm_paths(self)
        plt.close()
        return BM 

        
    def gbm_paths(self,mu,sigma):
        N = self.N; paths = self.paths; S_0 = self.S_0
        
        BM = Simulations.get_bm_array(self)
        global GBM
        GBM = np.zeros((N,paths))
        GBM[0,:] = np.array([S_0 for i in range(paths)])
        
        for i in range(paths):
            GBM[1:N,i] = S_0*np.exp(sigma*BM[1:N,i] + (mu-0.5*sigma**2)*t[1:N])
        
        plt.plot(t,GBM) 
        plt.title("GBM paths simulation") 
        plt.xlabel("t")
        plt.ylabel("S(t)")
        
        
    def get_gbm_array(self,mu,sigma):
        Simulations.gbm_paths(self,mu,sigma)
        plt.close()
        return GBM     
  
        
    def poisson_paths(self,lamda):
        N = self.N; paths = self.paths
        
        global Nt
        Nt = np.zeros((N,paths))
        Nt[0,:] = np.array([0 for i in range(paths)])
        
        for i in range(paths):
            Nt[1:N,i] = np.cumsum(np.random.poisson(lam=lamda*(1/N),size=N-1))
           
        plt.plot(t,Nt)    
        plt.title("Poisson process paths simulation")
        plt.xlabel("t")
        plt.ylabel("N(t)")
        
    def get_poisson_array(self,lamda):
        Simulations.poisson_paths(self,lamda)
        plt.close()
        return Nt

    
    def mjd_paths(self,mu,sigma,lamda,m,delta):
        N = self.N; paths = self.paths
        
        GBM = Simulations.get_gbm_array(self,mu,sigma)
        
        Nt = Simulations.get_poisson_array(self, lamda)
        Nt_inc = Nt[1:N,:] - Nt[0:N-1,:]
        
        jumps = np.zeros((N,paths))
        jumps[0,:] = np.array([0 for i in range(paths)])
        global MJD
        MJD = np.zeros((N,paths))
        
        for i in range(paths):
            temp = []
            for j in Nt_inc[:,i]:
                if j == 1:
                    temp.append(np.random.normal(m,delta))
                else: temp.append(0)
            jumps[1:N,i] = np.cumsum(temp) 
        
        MJD = GBM*np.exp(jumps)
        
        plt.plot(t,MJD)
        plt.xlabel('t') 
        plt.ylabel('S(t)')
        plt.title('MJD paths simulation')
        
    def get_mjd_array(self,mu,sigma,lamda,m,delta):
        Simulations.mjd_paths(self, mu, sigma, lamda, m, delta)
        plt.close()
        return MJD 
    
    
    def kjd_paths(self,mu,sigma,lamda,p,neta_1,neta_2):
        N = self.N; paths = self.paths
        
        GBM = Simulations.get_gbm_array(self,mu,sigma)
        
        Nt = Simulations.get_poisson_array(self, lamda)
        Nt_inc = Nt[1:N,:] - Nt[0:N-1,:]
        
        jumps = np.zeros((N,paths))
        jumps[0,:] = np.array([0 for i in range(paths)])
        global KJD
        KJD = np.zeros((N,paths))
        
        for i in range(paths):
            temp = []
            for j in Nt_inc[:,i]:
                if j == 1:
                    if np.random.binomial(1,p) == 1:
                       temp.append(np.random.exponential(scale=1/neta_1))
                    else : temp.append(-np.random.exponential(scale=1/neta_2))
                else: temp.append(0)
            jumps[1:N,i] = np.cumsum(temp) 
        
        KJD = GBM*np.exp(jumps)
        
        plt.plot(t,KJD)
        plt.xlabel('t') 
        plt.ylabel('S(t)')
        plt.title('KJD paths simulation')
        
        
    def get_kjd_array(self,mu,sigma,lamda,p,neta_1,neta_2):
        Simulations.kjd_paths(self, mu, sigma, lamda, p, neta_1, neta_2)
        plt.close()
        return KJD
        
       
        
   
        
