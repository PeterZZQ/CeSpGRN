import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KELLER_ori(nn.Module):
    def __init__(self, X, K, lr = 1e-3, seed = 0, theta_init = None):
        super(KELLER_ori, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)


        # shape (ntimes, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.lr = lr

        self.diag_mask = torch.eye(self.X.shape[1]).to(device)

        # shape (ngenes, ngenes) 
        if theta_init is None:   
            theta = (1 - self.diag_mask) * torch.randn(self.X.shape[1], self.X.shape[1]).to(device)
            self.theta = nn.Parameter(theta)
        else:
            self.theta = nn.Parameter(torch.FloatTensor(theta_init))

        self.opt = Adam(self.parameters(), lr = self.lr)
        self.diag_mask = torch.eye(self.theta.shape[0]).to(device)
        # shape (ntimes, ntimes)
        self.kernel = ((np.ones(X.shape[0])/X.shape[0])*K)/np.sum(K,axis=0)
        self.kernel = torch.FloatTensor(self.kernel).to(device)
        
        self.best_loss = 1e6

    def pseudo_lkl_loss(self, idx):
        if idx is not None:
            #thetax: (ntimes, gene) = (ntimes, ngenes) @ (ngenes, gene)
            thetax = self.X @ (self.theta - self.diag_mask * self.theta)[idx:(idx+1),:].t()

            #gamma: (ntimes, gene) = (ntimes, gene) * (ntimes, gene)
            gamma = thetax * self.X[:,idx:(idx+1)] - torch.log(torch.exp(thetax) + torch.exp(-thetax))

        else:
            #thetax: (ntimes, ngenes) = (ntimes, ngenes) @ (ngenes, ngenes)
            thetax = self.X @ (self.theta - self.diag_mask * self.theta).t()

            #gamma: (ntimes, ngenes) = (ntimes, ngenes) * (ntimes, ngenes)
            gamma = thetax * self.X - torch.log(torch.exp(thetax) + torch.exp(-thetax))
        return gamma

    def train(self, t, n_iter = 1000, n_intervals = 1000, lamb = 2.1e-4):
        count = 0
        k = self.kernel[t,:][:,None]

        for it in range(n_iter):
            # loop through all genes
            for u in range(self.X.shape[1]): 
                for v in range(self.X.shape[1]): 
                    self.opt.zero_grad()
                    loss1 = - (k * self.pseudo_lkl_loss(u)).sum()
                    loss2 = (self.theta - self.diag_mask * self.theta)[u:(u+1),:].abs().sum()

                    loss = 1000 * loss1 + lamb * loss2
                    loss.backward()

                    # make the gradient of only (u,v) exist
                    self.theta.grad.data[u,:v].fill_(0)
                    self.theta.grad.data[u,(v+1):].fill_(0)
                    self.opt.step()

            # Validation
            if (it + 1) % n_intervals == 0:
                with torch.no_grad():
                    loss1 = - 1000 * (k * self.pseudo_lkl_loss(idx = None)).sum() 
                    loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
                    valid_loss = loss1 + loss2
                print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
                info = [
                    'negative pseudo lkl: {:.5f}'.format(loss1.item()),
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



class KELLER_vec(nn.Module):
    def __init__(self, X, K, lr = 1e-3, seed = 0, theta_init = None):
        super(KELLER_vec, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.lr = lr

        self.diag_mask = torch.eye(self.X.shape[1]).to(device)

        # shape (ngenes, ngenes) 
        if theta_init is None:   
            theta = (1 - self.diag_mask) * torch.randn(self.X.shape[1], self.X.shape[1]).to(device)
            self.theta = nn.Parameter(theta)
        else:
            self.theta = nn.Parameter(torch.FloatTensor(theta_init))

        self.opt = Adam(self.parameters(), lr = self.lr)
        # shape (ntimes, ntimes)
        self.kernel = ((np.ones(X.shape[0])/X.shape[0])*K)/np.sum(K,axis=0)
        self.kernel = torch.FloatTensor(self.kernel).to(device)
        
        self.best_loss = 1e6

    def pseudo_lkl_loss(self, idx):
        if idx is not None:
            #thetax: (ntimes, gene) = (ntimes, ngenes) @ (ngenes, gene)
            thetax = self.X @ (self.theta - self.diag_mask * self.theta)[idx:(idx+1),:].t()

            #gamma: (ntimes, gene) = (ntimes, gene) * (ntimes, gene)
            gamma = thetax * self.X[:,idx:(idx+1)] - torch.log(torch.exp(thetax) + torch.exp(-thetax))

        else:
            #thetax: (ntimes, ngenes) = (ntimes, ngenes) @ (ngenes, ngenes)
            thetax = self.X @ (self.theta - self.diag_mask * self.theta).t()

            #gamma: (ntimes, ngenes) = (ntimes, ngenes) * (ntimes, ngenes)
            gamma = thetax * self.X - torch.log(torch.exp(thetax) + torch.exp(-thetax))
        return gamma

    def train(self, t, n_iter = 2000, n_intervals = 1000, lamb = 2.1e-4):
        count = 0
        k = self.kernel[t,:][:,None]

        for it in range(n_iter):
            # loop through all genes
            for idx in range(self.X.shape[1]):  
                self.opt.zero_grad()
                loss1 = - (k * self.pseudo_lkl_loss(idx)).sum()
                loss2 = (self.theta - self.diag_mask * self.theta)[idx:(idx+1),:].abs().sum()

                # (ntimes,1) weight vector
                loss = 1000 * loss1 + lamb * loss2
                loss.backward()

                self.theta.grad.data[:idx,:].fill_(0)
                self.theta.grad.data[(idx+1):,:].fill_(0)

                self.opt.step()

            # Validation
            if (it + 1) % n_intervals == 0:
                with torch.no_grad():
                    loss1 = - 1000 * (k * self.pseudo_lkl_loss(idx = None)).sum() 
                    loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
                    valid_loss = loss1 + loss2
                print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
                info = [
                    'negative pseudo lkl: {:.5f}'.format(loss1.item()),
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



class KELLER_mat(nn.Module):
    """\
        Formula can be parallelized, still fit in the pseudo-likelihood purpose??
    """
    def __init__(self, X, K, lr = 1e-3, seed = 0, theta_init = None):
        super(KELLER_mat, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.lr = lr

        self.diag_mask = torch.eye(self.X.shape[1]).to(device)

        # shape (ngenes, ngenes) 
        if theta_init is None:   
            theta = (1 - self.diag_mask) * torch.randn(self.X.shape[1], self.X.shape[1]).to(device)
            self.theta = nn.Parameter(theta)
        else:
            self.theta = nn.Parameter(torch.FloatTensor(theta_init))

        self.opt = Adam(self.parameters(), lr = self.lr)

        # shape (ntimes, ntimes)
        self.kernel = torch.FloatTensor(K).to(device)
        self.num_sample = self.X.shape[0]//self.kernel.shape[0]
        
        self.best_loss = 1e6

    def pseudo_lkl_loss(self):
        #thetax: (ntimes, ngenes) = (ntimes, ngenes) @ (ngenes, ngenes)
        thetax = self.X @ (self.theta - self.diag_mask * self.theta).t()

        #gamma: (ntimes, ngenes) = (ntimes, ngenes) * (ntimes, ngenes)
        gamma = thetax * self.X - torch.log(torch.exp(thetax) + torch.exp(-thetax))
        return gamma

    def train(self, t, n_iter = 5000, n_intervals = 1000, lamb = 2.1e-4):
        count = 0
        k = self.kernel[t,:][:,None]
        w = k.repeat(1,self.num_sample).reshape(-1,1)
        assert (w.sum() - self.num_sample).abs() < 1e-4

        for it in range(n_iter):  
            self.opt.zero_grad()
            loss1 = - (w * self.pseudo_lkl_loss()).sum()
            loss2 = (self.theta - self.diag_mask * self.theta).abs().sum()

            # (ntimes,1) weight vector
            loss = 1000 * loss1 + lamb * loss2
            loss.backward()
            self.opt.step()

            # Validation
            if (it + 1) % n_intervals == 0:
                with torch.no_grad():
                    loss1 = - 1000 * (w * self.pseudo_lkl_loss()).sum() 
                    loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
                    valid_loss = loss1 + loss2
                print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
                info = [
                    'negative pseudo lkl: {:.5f}'.format(loss1.item()),
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


class KELLER_mat2(nn.Module):
    """\
        Formula can be parallelized, still fit in the pseudo-likelihood purpose??
    """
    def __init__(self, X, K, lr = 1e-3, seed = 0, theta_init = None):
        super(KELLER_mat2, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, ngenes)
        self.X = torch.FloatTensor(X).to(device)
        self.lr = lr

        self.diag_mask = torch.eye(self.X.shape[1]).to(device)

        # shape (ngenes, ngenes) 
        if theta_init is None:   
            theta = (1 - self.diag_mask) * torch.randn(self.X.shape[1], self.X.shape[1]).to(device)
            self.theta = nn.Parameter(theta)
        else:
            self.theta = nn.Parameter(torch.FloatTensor(theta_init))

        self.opt = Adam(self.parameters(), lr = self.lr)

        # shape (ntimes, ntimes)
        self.kernel = torch.FloatTensor(K).to(device)
        self.num_sample = self.X.shape[0]//self.kernel.shape[0]
        
        self.best_loss = 1e6

    def pseudo_lkl_loss(self):
        #thetax: (ntimes, ngenes) = (ntimes, ngenes) @ (ngenes, ngenes)
        thetax = self.X @ (self.theta - self.diag_mask * self.theta).t()

        #gamma: (ntimes, ngenes) = (ntimes, ngenes) * (ntimes, ngenes)
        gamma = thetax * self.X - torch.log(torch.exp(thetax) + torch.exp(-thetax))
        return gamma

    def train(self, t, n_iter = 5000, n_intervals = 1000, lamb = 2.1e-4):
        count = 0
        k = self.kernel[t,:][:,None]
        w = k.repeat(1,self.num_sample).reshape(-1,1)
        assert (w.sum() - self.num_sample).abs() < 1e-4

        for it in range(n_iter):  
            self.opt.zero_grad()
            loss1 = - (w * self.pseudo_lkl_loss()).sum()
            # loss2 = (self.theta - self.diag_mask * self.theta).abs().sum()

            # (ntimes,1) weight vector
            loss = loss1
            loss.backward()
            self.opt.step()

            # soft-thresholding
            with torch.no_grad():
                self.theta.data = (self.theta >= lamb * self.lr) * (self.theta - lamb * self.lr) + (self.theta <= - lamb * self.lr) * (self.theta + lamb * self.lr)

            # # Validation
            # if (it + 1) % n_intervals == 0:
            #     with torch.no_grad():
            #         loss1 = - (w * self.pseudo_lkl_loss()).sum() 
            #         loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
            #         valid_loss = loss1 + loss2
            #     print("n_iter: {}, loss: {:.8f}".format(it+1, valid_loss.item()))
                
            #     info = [
            #         'negative pseudo lkl: {:.5f}'.format(loss1.item()),
            #         'sparsity loss: {:.5f}'.format(loss2.item())
            #     ]
            #     for i in info:
            #         print("\t", i)
                
            #     if valid_loss.item() >= self.best_loss:
            #         count += 1
            #         if count % 10 == 0:
            #             self.optimizer.param_groups[0]['lr'] *= 0.5
            #             print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
            #             if self.optimizer.param_groups[0]['lr'] < 1e-6:
            #                 break
            
        return self.theta.cpu().detach().numpy()


class KELLER_batch(nn.Module): # working on...
    """\
        Formula can be parallelized, still fit in the pseudo-likelihood purpose??
    """
    def __init__(self, X, K, lr = 1e-3, seed = 0, theta_init = None):
        super(KELLER_batch, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, nsamples, ngenes)

        self.X = torch.FloatTensor(X).to(device)
        self.ntimes, self.nsamples, self.ngenes = self.X.shape

        # reshape (ntimes * nsamples, ngenes)
        self.X = self.X.reshape(self.ntimes*self.nsamples,self.ngenes) 
        self.lr = lr

        # shape (ntimes, ngenes, ngenes)
        self.diag_mask = torch.eye(self.ngenes).expand(self.ntimes*self.nsamples, self.ngenes, self.ngenes).to(device)

        # shape (ntimes, ngenes, ngenes)
        if theta_init is None:   
            theta = (1 - self.diag_mask) * torch.randn(self.diag_mask.shape).to(device)
            self.theta = nn.Parameter(theta)
        else:
            self.theta = nn.Parameter(torch.FloatTensor(theta_init))

        self.opt = Adam(self.parameters(), lr = self.lr)

        # shape (ntimes, ntimes)
        self.weights = torch.FloatTensor(K).to(device)

        # shape (total input dim(=ntimes), ntimes, ngnes,ngens) > (total input dim(=ntimes), ngnes,ngens) "weighted avg."
        assert torch.all(torch.sum(self.weights,dim=1) - 1 < 1e-6)
        
        self.best_loss = 1e6

    def pseudo_lkl_loss(self):
        #thetax: (ntimes * nsamples, ngenes) = (ntimes * nsamples, ngenes) @ (ngenes, ngenes)
        thetax = self.X @ torch.transpose(self.theta - self.diag_mask * self.theta, 1, 2)

        #gamma: (ntimes * nsamples, ngenes) = (ntimes * nsamples, ngenes) * (ntimes * nsamples, ngenes) - (ntimes * nsamples, ngenes)
        gamma = thetax * self.X - torch.log(torch.exp(thetax) + torch.exp(-thetax))
        return gamma

    def train(self, n_iter = 5000, n_intervals = 1000, lamb = 2.1e-4):
        count = 0
#        k = self.kernel[t,:][:,None]
#        w = k.repeat(1,self.num_sample).reshape(-1,1)
#        assert (w.sum() - self.num_sample).abs() < 1e-4

        for it in range(n_iter):  
            self.opt.zero_grad()
            loss1 = self.weights[:,:,None,None,None] * self.pseudo_lkl_loss().reshape(self.ntimes,self.nsamples,self.ntimes*self.nsamples,self.ngenes)[None,:,:,:,:]
            loss1 = loss1.reshape(self.ntimes,self.ntimes*self.nsamples,self.ntimes*self.nsamples,self.ngenes)
            loss1 = -torch.sum(loss1, dim=1)
            # loss2 = (self.theta - self.diag_mask * self.theta).abs().sum()

            # (ntimes,1) weight vector
            loss = torch.sum(loss1)
            loss.backward()
            self.opt.step()

            print("iter: ",it)

            # soft-thresholding
            with torch.no_grad():
                self.theta.data = (self.theta >= lamb * self.lr) * (self.theta - lamb * self.lr) + (self.theta <= - lamb * self.lr) * (self.theta + lamb * self.lr)

            # Validation
            if (it + 1) % n_intervals == 0:
                with torch.no_grad():
                    loss1 = -torch.sum(self.weights[:,:,None,None] * self.pseudo_lkl_loss()[None,:,:,:], dim=1)
                    loss2 = lamb * (self.theta - self.diag_mask * self.theta).abs().sum()
                    valid_loss = loss1 + loss2
                print("n_iter: {}, loss: {:.8f}".format(it+1, torch.sum(valid_loss).item()))
                
                info = [
                    'negative pseudo lkl: {:.5f}'.format(torch.sum(loss1).item()),
                    'sparsity loss: {:.5f}'.format(torch.sum(loss2).item())
                ]
                for i in info:
                    print("\t", i)
                
                if torch.sum(valid_loss).item() >= self.best_loss:
                    count += 1
                    if count % 10 == 0:
                        self.opt.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(it + 1, self.opt.param_groups[0]['lr']))
                        if self.opt.param_groups[0]['lr'] < 1e-6:
                            break
            
        return self.theta.cpu().detach().numpy()

