import sys, os
sys.path.append('./')
import numpy as np
import torch
from torch_sqrtm import MatrixSquareRoot

from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_sqrtm = MatrixSquareRoot.apply

# ADMM
class G_admm():
    def __init__(self, X, K, TF = None, seed = 0):
        super(G_admm, self).__init__()
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


    def train(self, t, max_iters = 50, n_intervals = 1, lamb = 2.1e-4, rho = 1, beta = 0, theta_init_offset = 0.1):
        # kernel function
        w = self.weights[t,:][:,None, None]
        
        """
        import matplotlib.pyplot as plt
        plt.plot(w.cpu().numpy().squeeze())
        plt.show()
        """

        assert (w.sum() - 1).abs() < 1e-6
        # weight average of empirical covariance matrix, [ngenes, ngenes]
        empir_cov_ave = torch.sum(w * self.empir_cov, dim = 0)
        # initialize
        Z = torch.diag_embed(1/(torch.diagonal(empir_cov_ave, offset=0, dim1=-2, dim2=-1) + theta_init_offset))
        # positive definite matrix
        ll = torch.cholesky(Z)
        Z = torch.matmul(ll, ll.transpose(-1, -2))
        self.theta = Z
        
        U = torch.zeros(Z.shape).to(device)
        I = torch.eye(Z.shape[0]).to(device)
        zero = torch.Tensor([0]).to(device)

        it = 0
        loss1 = self.neg_lkl_loss(self.theta, empir_cov_ave)
        loss2 = (self.theta - self.diag_mask * self.theta).abs().sum()
        loss3 = torch.norm(self.mask * self.theta)
        print("Initial: loss1: {:.5f}, loss2: {:.5f}, loss3: {:.5f}".format(loss1.item(), loss2.item(), loss3.item()))


        while(it < max_iters):   
            # Y of the shape: (ngenes, ngenes)
            Y = U - Z + empir_cov_ave/rho
            self.theta = - 0.5 * Y + torch_sqrtm(Y.t() @ Y * 0.25 + I/rho)
            Z = torch.sign(self.theta + U) * torch.max((rho * (self.theta + U).abs() - lamb)/(rho + beta * self.mask), zero)
            U = U + self.theta - Z
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


# ADMM tensor: data shape (ntimes, ngenes, ngenes)
class G_admm_batch():
    def __init__(self, X, K, TF = None, seed = 0, method = 0, pre_cov = None):
        super(G_admm_batch, self).__init__()
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
        # self.kernel = ((np.ones(self.ntimes)/self.ntimes)*K)/np.sum(K,axis=0)
        self.weights = torch.FloatTensor(K).to(device)

        # shape (total input dim(=ntimes), ntimes, ngnes,ngens) > (total input dim(=ntimes), ngnes,ngens) "weighted avg."
        assert torch.all(torch.sum(self.weights,dim=1) - 1 < 1e-6)
        self.w_empir_cov = torch.sum((self.weights[:,:,None,None]*self.empir_cov[None,:,:,:]),dim=1)

        # method 1 (cal entire sample covariance matrix > assigning weights)
        self.X_all = self.X.reshape(1, self.ntimes * self.nsamples, self.ngenes)
        self.epir_mean_all = self.X_all.mean(dim = 1, keepdim = True).expand(1, self.ntimes * self.nsamples, self.ngenes)
        X_all = self.X_all - self.epir_mean_all
        self.empir_cov_all = torch.bmm(X_all.reshape((self.ntimes * self.nsamples, self.ngenes, 1)), X_all.reshape((self.ntimes * self.nsamples, 1, self.ngenes)))
        
        self.empir_cov_all = torch.sum(self.empir_cov_all.reshape(self.ntimes, self.nsamples, self.ngenes, self.ngenes), dim=1)/(self.nsamples-1)
        self.w_empir_cov_all = torch.sum((self.weights[:,:,None,None] * self.empir_cov_all[None,:,:,:]), dim=1)

        if method == 1:
            self.w_empir_cov = self.w_empir_cov_all
        
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


    def train(self, max_iters = 50, n_intervals = 1, lamb = 2.1e-4, rho = 1, beta = 0, theta_init_offset = 0.1):
        
       # initialize
        Z = torch.diag_embed(1/(torch.diagonal(self.w_empir_cov, offset=0, dim1=-2, dim2=-1) + theta_init_offset))
        # positive definite matrix
        ll = torch.linalg.cholesky(Z)
        Z = torch.matmul(ll, ll.transpose(-1, -2))
        self.thetas = Z
        
        U = torch.zeros(Z.shape).to(device)
        I = torch.eye(self.ngenes).expand(Z.shape).to(device)

        it = 0
        loss1 = self.neg_lkl_loss(self.thetas, self.w_empir_cov).sum()
        loss2 = (self.thetas - self.diag_mask * self.thetas).abs().sum()
        loss3 = torch.norm(self.mask * self.thetas)
        print("Initial: loss1: {:.5f}, loss2: {:.5f}, loss3: {:.5f}".format(loss1.item(), loss2.item(), loss3.item()))

        while(it < max_iters):   
            # Y of the shape: (ntimes, ngenes, ngenes)
            Y = U - Z + self.w_empir_cov/rho

            self.thetas = - 0.5 * Y + torch.stack([torch_sqrtm(mat) for mat in (torch.transpose(Y,1,2) @ Y * 0.25 + I/rho)])
            Z = torch.sign(self.thetas + U) * torch.max((rho * (self.thetas + U).abs() - lamb)/(rho + beta * self.mask), torch.Tensor([0]).to(device))
            U = U + self.thetas - Z

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
