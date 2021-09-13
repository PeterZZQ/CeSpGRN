import sys, os
sys.path.append('./')

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch_sqrtm import MatrixSquareRoot
from g_ista import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_sqrtm = MatrixSquareRoot.apply

# GLAD
class G_glad():
    def __init__(self, X, K, TF = None, seed = 0, theta_init = None, pre_cov = None):
        super(G_glad, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, nsamples, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.ntimes, self.nsamples, self.ngenes = self.X.shape

        # Calculate the time series empirical covariance matrix
        if pre_cov is None:
            # shape (ntimes, nsamples, ngenes)
            self.epir_mean = self.X.mean(dim = 1, keepdim = True).expand(self.ntimes, self.nsamples, self.ngenes)
            X = self.X - self.epir_mean
            # (ntimes * nsamples, ngenes, ngenes)
            self.empir_cov = torch.bmm(X.reshape((self.ntimes * self.nsamples, self.ngenes, 1)), X.reshape((self.ntimes * self.nsamples, 1, self.ngenes)))
            # (ntimes, ngenes, ngenes)
            self.empir_cov = torch.sum(self.empir_cov.reshape((self.ntimes, self.nsamples, self.ngenes, self.ngenes)), dim = 1)/(self.nsamples - 1)
        else:
            self.empir_cov = pre_cov.to(device)
        
        self.diag_mask = torch.eye(self.ngenes).to(device)

        # shape (ntimes, ntimes)
        # self.kernel = ((np.ones(self.ntimes)/self.ntimes)*K)/np.sum(K,axis=0)
        self.weights = torch.FloatTensor(K).to(device)

        # mask matrix
        self.mask = 0
        if TF is not None:
            self.mask = torch.zeros(self.ngenes, self.ngenes).to(device)
            # mark probable interactions
            self.mask[TF, :] = 1
            self.mask[:, TF] = 1
            # element-wise reverse
            self.mask = 1 - self.mask     

    @staticmethod
    def neg_lkl_loss(theta, S):
        """\
        Description:
        --------------
            The negative log likelihood function
        Parameters:
        --------------
            theta:
                The estimated theta
            S:
                The empirical covariance matrix
        Return:
        --------------
            The negative log likelihood value
        """
        t1 = -1*torch.logdet(theta)
        t2 = torch.trace(torch.matmul(S, theta))
        return t1 + t2

    def train(self, t, max_iters = 500, n_intervals = 10, lamb = 2.1e-4, rho = 1, beta = 0, theta_init_offset = 0.1):
        # k of the shape: (ntimes, 1, 1)
        w = self.weights[t,:][:,None, None]

        assert (w.sum() - 1).abs() < 1e-6

        empir_cov_ave = torch.sum(w * self.empir_cov, dim = 0)
        # initialize
        Z = torch.diag_embed(1/(torch.diagonal(empir_cov_ave, offset=0, dim1=-2, dim2=-1) + theta_init_offset))
        ll = torch.cholesky(Z)
        Z = torch.matmul(ll, ll.transpose(-1, -2))
        self.theta = Z

        I = torch.eye(Z.shape[0]).to(device)
        zero = torch.Tensor([0]).to(device)

        it = 0
        loss1 = self.neg_lkl_loss(self.theta, empir_cov_ave)
        loss2 = (self.theta - self.diag_mask * self.theta).abs().sum()
        loss3 = torch.norm(self.mask * self.theta)
        print("Initial: loss1: {:.5f}, loss2: {:.5f}, loss3: {:.5f}".format(loss1.item(), loss2.item(), loss3.item()))
 
        while(it < max_iters):  
            # Y of the shape: (ngenes, ngenes)
            Y = empir_cov_ave/rho - Z            
            self.theta = - 0.5 * Y + torch_sqrtm(Y.t() @ Y * 0.25 + I/rho + 1e-6)
            # Z = torch.sign(self.theta) * torch.max((self.theta).abs() - lamb/rho, zero)
            Z = torch.sign(self.theta) * torch.max((rho * (self.theta).abs() - lamb)/(rho + beta * self.mask), zero)
            assert self.theta.shape == (self.ngenes, self.ngenes)

            # Validation
            if (it + 1) % n_intervals == 0:
                loss1 = self.neg_lkl_loss(self.theta, empir_cov_ave)
                loss2 = (self.theta - self.diag_mask * self.theta).abs().sum()
                loss3 = torch.norm(self.mask * self.theta)
                valid_loss = loss1 + lamb * loss2 + beta * loss3
                print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
                info = [
                    'negative log lkl: {:.5f}'.format(loss1.item()),
                    'sparsity loss: {:.5f}'.format(loss2.item()),
                    'TF knowledge loss: {:.5f}'.format(loss3.item())
                ]
                for i in info:
                    print("\t", i)
            it += 1

        return (self.theta * (1 - self.diag_mask)).data.cpu().numpy()
        # return self.theta.data.cpu().numpy()


# GLAD tensor: data shape (ntimes, ngenes, ngenes)
class G_glad_batch():
    def __init__(self, X, K, TF = None, seed = 0, theta_init = None, pre_cov = None):
        super(G_glad_batch, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, nsamples, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.ntimes, self.nsamples, self.ngenes = self.X.shape

        # TODO: consider the mean for variance calculation
        if pre_cov is None:
            # shape (ntimes, nsamples, ngenes)
            self.epir_mean = self.X.mean(dim = 1, keepdim = True).expand(self.ntimes, self.nsamples, self.ngenes)
            X = self.X - self.epir_mean
            # (ntimes * nsamples, ngenes, ngenes)
            self.empir_cov = torch.bmm(X.reshape((self.ntimes * self.nsamples, self.ngenes, 1)), X.reshape((self.ntimes * self.nsamples, 1, self.ngenes)))
            # (ntimes, ngenes, ngenes)
            self.empir_cov = torch.sum(self.empir_cov.reshape((self.ntimes, self.nsamples, self.ngenes, self.ngenes)), dim = 1)/(self.nsamples - 1)
        else:
            self.empir_cov = pre_cov

        self.diag_mask = torch.eye(self.ngenes).expand(self.empir_cov.shape).to(device)

        # shape (ntimes, ntimes)
        self.weights = torch.FloatTensor(K).to(device)

        # shape (total input dim(=ntimes), ntimes, ngnes,ngens) > (total input dim(=ntimes), ngnes,ngens) "weighted avg."
        assert torch.all(torch.sum(self.weights,dim=1) - 1 < 1e-6)
        self.w_empir_cov = torch.sum((self.weights[:,:,None,None]*self.empir_cov[None,:,:,:]),dim=1)

        # mask matrix (ntimes, ngenes, ngenes)
        self.mask = 0
        if TF is not None:
            self.mask = torch.zeros(self.ngenes, self.ngenes).to(device)
            # mark probable interactions
            self.mask[TF, :] = 1
            self.mask[:, TF] = 1
            # element-wise reverse
            self.mask = 1 - self.mask
            self.mask = self.mask.expand(self.ntimes, self.ngenes, self.ngenes)   
        
    @staticmethod
    def neg_lkl_loss(thetas, S):
        """\
        Description:
        --------------
            The negative log likelihood function
        Parameters:
        --------------
            theta:
                The estimated theta
            S:
                The empirical covariance matrix
        Return:
        --------------
            The negative log likelihood value
        """

        t1 = -1*torch.logdet(thetas)
        t2 = torch.stack([torch.trace(mat) for mat in torch.bmm(S, thetas)])

        return t1 + t2

    def train(self, max_iters = 500, n_intervals = 10, lamb = 2.1e-4, rho = 1, beta = 0, theta_init_offset = 0.1):

        # initialize
        Z = torch.diag_embed(1/(torch.diagonal(self.w_empir_cov, offset=0, dim1=-2, dim2=-1) + theta_init_offset))
        # positive definite matrix
        ll = torch.cholesky(Z)
        Z = torch.matmul(ll, ll.transpose(-1, -2))
        self.thetas = Z
        
        I = torch.eye(self.ngenes).expand(Z.shape).to(device)

        it = 0
        loss1 = self.neg_lkl_loss(self.thetas, self.w_empir_cov).sum()
        loss2 = (self.thetas - self.diag_mask * self.thetas).abs().sum()
        loss3 = torch.norm(self.mask * self.thetas)
        print("Initial: loss1: {:.5f}, loss2: {:.5f}, loss3: {:.5f}".format(loss1.item(), loss2.item(), loss3.item()))

        while(it < max_iters):   
            # Y of the shape: (ntimes, ngenes, ngenes)
            Y = self.w_empir_cov/rho - Z

            self.thetas = - 0.5 * Y + torch.stack([torch_sqrtm(mat) for mat in (torch.transpose(Y,1,2) @ Y * 0.25 + I/rho + 1e-6)])
            # Z = torch.sign(self.thetas) * torch.max(((self.thetas).abs() - lamb/rho), torch.Tensor([0]).to(device))
            Z = torch.sign(self.thetas) * torch.max((rho * (self.thetas).abs() - lamb)/(rho + beta * self.mask), torch.Tensor([0]).to(device))
            assert self.thetas.shape == (self.ntimes, self.ngenes, self.ngenes)

            # Validation
            if (it + 1) % n_intervals == 0:
                loss1 = self.neg_lkl_loss(self.thetas, self.w_empir_cov).sum()
                loss2 = (self.thetas - self.diag_mask * self.thetas).abs().sum()
                loss3 = torch.norm(self.mask * self.thetas)
                valid_loss = loss1 + lamb * loss2 + beta * loss3
                print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
                info = [
                    'negative log lkl: {:.5f}'.format(loss1.item()),
                    'sparsity loss: {:.5f}'.format(loss2.item()),
                    'TF knowledge loss: {:.5f}'.format(loss3.item())
                ]
                for i in info:
                    print("\t", i)
            it += 1
            
        return (self.thetas * (1 - self.diag_mask)).data.cpu().numpy()
        # return self.theta.data.cpu().numpy()


# vanilla gradient descent
class G_glad_grad():
    def __init__(self, X, K, lr = 1e-3, seed = 0, theta_init = None):
        super(G_glad_grad, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, nsamples, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.ntimes, self.nsamples, self.ngenes = self.X.shape

        # TODO: consider the mean for variance calculation
        # shape (ntimes, nsamples, ngenes)
        self.epir_mean = self.X.mean(dim = 1, keepdim = True).expand(self.ntimes, self.nsamples, self.ngenes)
        X = self.X - self.epir_mean
        # (ntimes * nsamples, ngenes, ngenes)
        self.empir_cov = torch.bmm(X.reshape((self.ntimes * self.nsamples, self.ngenes, 1)), X.reshape((self.ntimes * self.nsamples, 1, self.ngenes)))
        # (ntimes, ngenes, ngenes)
        self.empir_cov = torch.sum(self.empir_cov.reshape((self.ntimes, self.nsamples, self.ngenes, self.ngenes)), dim = 1)/(self.nsamples - 1)

        self.lr = lr

        self.diag_mask = torch.eye(self.ngenes).to(device)

        # shape (ngenes, ngenes) 
        if theta_init is None:   
            theta = (1 - self.diag_mask) * torch.randn(self.ngenes, self.ngenes).to(device)
            # make symmetric
            theta = (theta + theta.t()) * 0.5
            self.theta = nn.Parameter(theta)
        else:
            self.theta = nn.Parameter(torch.FloatTensor(theta_init))

        # make PD
        ll = torch.cholesky(self.theta)
        self.theta = torch.matmul(ll, ll.transpose(-1, -2)).to(device)        
        self.opt = Adam(self.parameters(), lr = self.lr)

        # shape (ntimes, ntimes)
        self.weights = torch.FloatTensor(K).to(device)
        self.best_loss = 1e6

    @staticmethod
    def neg_lkl_loss(theta, S):
        """\
        Description:
        --------------
            The negative log likelihood function
        Parameters:
        --------------
            theta:
                The estimated theta
            S:
                The empirical covariance matrix
        Return:
        --------------
            The negative log likelihood value
        """
        t1 = -1*torch.logdet(theta)
        t2 = torch.trace(torch.matmul(S, theta))
        return t1 + t2

    def train(self, t, n_iter = 5000, n_intervals = 1000, lamb = 2.1e-4):
        count = 0
        w = self.weights[t,:][:,None]
        assert (w.sum() - 1).abs() < 1e-6

        for it in range(n_iter):  
            self.opt.zero_grad()
            loss1 = (w * self.neg_lkl_loss()).sum()
            loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
            loss = loss1 + loss2
            loss.backward()
            self.opt.step()

            # Validation
            if (it + 1) % n_intervals == 0:
                with torch.no_grad():
                    loss1 = self.neg_lkl_loss(self.theta, torch.sum(w * self.empir_cov, dim = 0))
                    loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
                    valid_loss = loss1 + loss2
                print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
                info = [
                    'negative log lkl: {:.5f}'.format(loss1.item()),
                    'sparsity loss: {:.5f}'.format(loss2.item())
                ]
                for i in info:
                    print("\t", i)
                
                if valid_loss.item() >= self.best_loss:
                    count += 1
                    if count % 10 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
        return self.theta

