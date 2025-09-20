import unittest
import numpy as np
import algorithms.SCoNE as SCoNE


class TestSCoNE(unittest.TestCase):
    def setUp(self):
        self.rank = 3
        self.W = np.random.random((2500, 3))*1e-2+1e-6 
        self.H_G = np.random.random((200, 3))*1e-2+1e-6
        self.H_C = np.random.random((100, 3))*1e-2+1e-6
        self.U_G = np.random.random((200, 10))*1e-2+1e-6
        self.U_C = np.random.random((100, 10))*1e-2+1e-6
    
    def test_none_loss_type(self):
        G = np.random.choice([0, 1], (2500,200))
        C = np.random.poisson(lam=3, size=(2500,100))
        # if C_loss_type and G_loss_type are None, shouldn't be any change in factor matrices from random
        factor_matrices_noz, loss_function_noz = SCoNE.alternating_opt(G,C,np.zeros((2500,10)),
            self.W, self.H_G, self.H_C, self.U_G, self.U_C,lambda_W=0, lambda_H_G=0, lambda_H_C=0, 
            method='L-BFGS-B',options={'maxcor':10, 'maxiter':10, 'gtol':1e-5, 'maxls':20, 'ftol':1e-6},tol=1e-4,nonneg=True, C_loss_type=None, G_loss_type=None)
