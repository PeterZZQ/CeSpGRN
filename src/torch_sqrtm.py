import torch
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import scipy.linalg

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
       Given a positive semi-definite matrix X,
       X = X^{1/2}X^{1/2}, compute the gradient: dX^{1/2} by solving the Sylvester equation, 
       dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
    """
    @staticmethod
    def forward(ctx, input):
        #m = input.numpy().astype(np.float_)
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real)#.type_as(input)
        ctx.save_for_backward(sqrtm) # save in cpu
        sqrtm = sqrtm.type_as(input)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            #sqrtm, = ctx.saved_variables
            sqrtm, = ctx.saved_tensors
            #sqrtm = sqrtm.data.numpy().astype(np.float_)
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            #gm = grad_output.data.numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
#            gm = np.eye(grad_output.shape[-1])
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return Variable(grad_input)


sqrtm = MatrixSquareRoot.apply

def original_main():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = k.t().matmul(k)
    pd_mat = Variable(pd_mat, requires_grad=True)
    test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    print(test)

def single_main():
    from torch.autograd import gradcheck 
    n = 1
    A = torch.randn( 20, 10).double()
    # Create a positive definite matrix
    pd_mat = A.t().matmul(A)
    pd_mat = Variable(pd_mat, requires_grad=True)
    test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    print(test)

    #sqrtm_scipy = np.zeros_like(A)
    print('err: ', pd_mat)
    sqrtm_scipy = scipy.linalg.sqrtm(pd_mat.detach().numpy().astype(np.float_))
#    for i in range(n):
#        sqrtm_scipy[i] = sqrtm(pd_mat[i].detach().numpy())
    sqrtm_torch = sqrtm(pd_mat)
    print('sqrtm torch: ', sqrtm_torch)
    print('scipy', sqrtm_scipy)
    print('Difference: ', np.linalg.norm(sqrtm_scipy - sqrtm_torch.detach().numpy()))

def main():# batch
    from torch.autograd import gradcheck 
    n = 2
    A = torch.randn(n, 4, 5).double()
    A.requires_grad = True
    # Create a positive definite matrix
    #pd_mat = A.t().matmul(A)
    pd_mat = torch.matmul(A.transpose(-1, -2), A)
    pd_mat = Variable(pd_mat, requires_grad=True)
    print('err: ', pd_mat.shape)
    #test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    #print(test)

    sqrtm_scipy = np.zeros_like(pd_mat.detach().numpy())
    #sqrtm_scipy = scipy.linalg.sqrtm(pd_mat.detach().numpy().astype(np.float_))
    for i in range(n):
        sqrtm_scipy[i] = scipy.linalg.sqrtm(pd_mat[i].detach().numpy())
    # batch implementation
    sqrtm_torch = torch.zeros(pd_mat.shape)
    for i in range(n):
        sqrtm_torch[i] = sqrtm(pd_mat[i])
    #sqrtm_torch = sqrtm(pd_mat)
    print('sqrtm torch: ', sqrtm_torch)
    print('scipy', sqrtm_scipy)
    print('Difference: ', np.linalg.norm(sqrtm_scipy - sqrtm_torch.detach().numpy()))

if __name__ == '__main__':
    main()
