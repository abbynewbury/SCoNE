import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao
import math
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

# Modeled after Huang et al. Flexible and Efficient Algorithmic Framework for Constrained Matrix and Tensor Factorization

def forward_substitution(L, b):
    """ Solve L * y = b for y (Forward Substitution) """
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def backward_substitution(L_T, y):
    """ Solve L.T * x = y for x (Backward Substitution) """
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(len(y) - 1, -1, -1):  # Start from last row
        x[i] = (y[i] - np.dot(L_T[i, i+1:], x[i+1:])) / L_T[i, i]
    return x


def admm_algorithm(Y, W, WTW,rank, H, U, constraints=None, orthogonality_constraints = False, e=0.01, calibration_steps=5, rho=None, plot=False): # remove PLOT -- for troubleshooting
    '''
    Y: for tensor case, mode d unfolding of tensor (for Kolda defn of unfolding this should be transpose of mode d unfolding)
    W: for tensor case, khatri-rao product of all modes not equal to d
    rank: rank of low-rank approx
    constraints: bound constraints of form [i,j,lower,upper] for np.clip
    orthogonality_constraints: force factor matrices to be orthogonal
    e: termination tolerance
    calibration_steps: number of iterations before considering termination
    '''
    initial_calibration_steps = calibration_steps

    # 2. G=W^TW - sped up by conversion to Hadamard product
    G = WTW

    #3. set p (empirically determined)
    if rho is None:
        rho = np.trace(G)/rank

    # 4. Calculate L from the Cholesky decomposition of G + rho*I
    L = np.linalg.cholesky(G+rho*np.identity(G.shape[0]))

    # 5. F = W^TY
    F = W.T@Y

    r = e+1 # probably a cleaner way to do this - needs to be this way due to transition to termination after calibration
    s = e+1
    max_iter = 20 # TODO: might want to fix or remove
    total_iters = 0 # TODO
    total_iters_r = [] # TODO: might want to fix or remove
    #6. calibration (# calibration steps determined by me - can be changed) then stopping param
    while ((calibration_steps>0) or (r > e or s > e)) and  ((calibration_steps>0) or (max_iter>0)):
        # cache H0 for calculation of S
        if calibration_steps == initial_calibration_steps:
            H0 = H

        # 7. use forward/back substitution to solve for H_tilde
        cols = []
        for col in range((F+rho*(H+U).T).shape[1]):
            y = forward_substitution(L, (F+rho*(H+U).T)[:,col])
            cols.append(backward_substitution(L.T, y))
        H_tilde = np.column_stack(cols)

        # 8. Update per constraints
        H_bar = H_tilde.T - U
        if orthogonality_constraints is True or constraints is not None:
            # add in orthogonality constraint 
            if orthogonality_constraints is True:
                U_bar, E_bar, V_T_bar = np.linalg.svd(H_bar, full_matrices=False)
                H = U_bar@V_T_bar
            else:
                H = H_bar

            # # add in orthogonality constraint -- TODO change to just enforce that columns sum to 1 for now - this forces r increasing as well
            # if orthogonality_constraints is True:
            #     col_sums = np.sum(H_bar, axis=0, keepdims=True) 
            #     H = H_bar - (np.ones((H_bar.shape[0], 1)) @ (col_sums - 1) / H_bar.shape[0])
            #     col_norms = np.linalg.norm(H, axis=0, keepdims=True) 
            #     H /= col_norms 
            # else:
            #     H = H_bar

            # add in elementwise bound constraints
            if constraints is not None: # add in constraints
                assert all([i[0] is not None for i in constraints]) and all([i[1] is not None for i in constraints]), "Need to provide i,j indices for constraint"

                constraints_array = np.array(constraints)
                rows, cols = constraints_array[:, 0].astype(int), constraints_array[:, 1].astype(int)
                lower_bounds, upper_bounds = constraints_array[:, 2], constraints_array[:, 3]
                H[rows, cols] = np.clip(H[rows, cols], lower_bounds, upper_bounds)

            # # add in vector orthogonality constraint
            # if constraints is not None:
            #     for constraint in constraints:
            #         j = constraint[0]
            #         k = constraint[1]
            #         orthogonal_vec = constraint[2]
            #         # update Hk
            #         column_update = H[:,k] - ((np.dot(orthogonal_vec, H[:,k])/np.dot(orthogonal_vec,orthogonal_vec))*orthogonal_vec)
            #         H[:,k] = column_update
            #         assert np.isclose(np.dot(H[:,k],orthogonal_vec),0)

            
            # # recheck orthogonality - not the issue currently since just performing SVD with no elementwise bound constraints
            # if orthogonality_constraints is True:
            #     if not (np.isclose(H.T@H,np.identity(H.shape[1]),atol=1e-8)).all():
            #         # re-assert constraint
            #         U_prime, E_prime, V_T_prime = np.linalg.svd(H, full_matrices=False)
            #         H = U_prime@V_T_prime
            #         assert (np.isclose(H.T@H,np.identity(H.shape[1]),atol=1e-8)).all()

        else: # no constraints
            H = H_bar


        # 9. Update U
        U = U + H - H_tilde.T

        # 10. Calibrate or stopping criteria
        max_iter -=1 # TODO

        if calibration_steps > 0:
            calibration_steps -= 1
        else: # set termination criterion
            # # if no change in r or s from last update
            # if r==np.linalg.norm(H-H_tilde.T)**2/np.linalg.norm(H)**2 or s==np.linalg.norm(H-H0)**2/np.linalg.norm(U)**2:
            #     print('breaking here')
            #     break
            r = np.linalg.norm(H-H_tilde.T)**2/np.linalg.norm(H)**2
            if np.linalg.norm(U)**2==0: # will happen with no constraints
                s = e-1 # to not have while loop dependent on s
            else:
                s = np.linalg.norm(H-H0)**2/np.linalg.norm(U)**2
        
        total_iters += 1 # TODO
        print(total_iters)
        total_iters_r.append([total_iters,r])
    if plot == True:
        total_iters_df = pd.DataFrame(total_iters_r, columns=['iteration','r'])
        total_iters_df = total_iters_df[total_iters_df['iteration']>5] # first 5 used for calibration
        plot = ggplot(total_iters_df,aes(x='iteration',y='r')) + geom_point() + geom_line()
        print(plot)
        plt.clf()

    # 11. return H and U
    print('returning')
    return H, U 



def constrained_cp_decomp(tensor, rank, A_indices = None, B_indices=None, D_indices=None, 
                            B_orth=False, D_orth=False, reconstruction_error_change_threshold=1e-4, e=0.01, verbose=True, calibration_steps=5, rho=None):
    '''
    rank: rank for low-rank decomposition
    A_B_non_sig_indices: indices j,k such that element j of B constrained to NOT be sig. associated with a_k (p-val < 5e-8)
    A_D_sig_indices: indices j,k such that element j of D constrained to NOT be sig. associated with a_k (p-val < 5e-8)
    '''
    # initializing H and U
    A = np.random.random((tensor.shape[0], rank))
    U_A = np.zeros((tensor.shape[0], rank))
    B = np.random.random((tensor.shape[1], rank))
    U_B = np.zeros((tensor.shape[1], rank))
    D = np.random.random((tensor.shape[2], rank))
    U_D = np.zeros((tensor.shape[2], rank))



    # cache certain matrix computations
    Y_cached = {0:f_unfold(tensor,0).T,1:f_unfold(tensor,1).T,2:f_unfold(tensor,2).T} 

    # repeat until relative change in reconstruction error less than reconstruction_error_change_threshold
    iterations = 0
    reconstruction_error = []
    relative_change_reconstruction_error = 1
    epochs = 0
    epochs_plot = [] # TODO
    while relative_change_reconstruction_error > reconstruction_error_change_threshold:
        
        # Update B
        if B_indices is not None: 
            mode_B_constraints = {'Bound':B_indices, 'Orthogonal':B_orth} 
        else:
            mode_B_constraints = {'Bound':None, 'Orthogonal':B_orth} 
        B,U_B = admm_algorithm(Y = Y_cached[1], W = khatri_rao([D, A]), WTW = np.multiply(D.T@D,A.T@A), rank = rank, H = B, U = U_B,
                             constraints = mode_B_constraints['Bound'], orthogonality_constraints=mode_B_constraints['Orthogonal'], calibration_steps=calibration_steps, e=e, rho=rho, plot=False) # TODO remove plot
        
        # Update D
        if D_indices is not None: 
            mode_D_constraints = {'Bound':D_indices, 'Orthogonal':D_orth}
        else:
            mode_D_constraints = {'Bound':None, 'Orthogonal':D_orth} 
        D,U_D = admm_algorithm(Y = Y_cached[2], W = khatri_rao([B, A]), WTW = np.multiply(B.T@B,A.T@A), rank = rank, H = D, U = U_D,
                             constraints = mode_D_constraints['Bound'], orthogonality_constraints=mode_D_constraints['Orthogonal'], calibration_steps=calibration_steps, e=e, rho=rho, plot=False) # TODO remove plot

        # Update A / TODO - see if it matters that we speed up calculation of WTY with Hadamard product here
        if A_indices is not None: 
            mode_A_constraints = {'Bound':A_indices, 'Orthogonal':D_orth} 
        else:
            mode_A_constraints = {'Bound':None, 'Orthogonal':D_orth} 
        A,U_A = admm_algorithm(Y = Y_cached[0], W = khatri_rao([D, B]), WTW = np.multiply(D.T@D,B.T@B), rank = rank, H = A, U = U_A,
                             constraints = mode_A_constraints['Bound'], orthogonality_constraints=False, calibration_steps=calibration_steps, e=e, rho=rho, plot=False)
        
        
        # calculate reconstruction error and relative change
        reconstructed = tl.cp_to_tensor((np.array([1]*rank), [A,B,D]))
        error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
        reconstruction_error.append(error)
        if len(reconstruction_error)>1:
            relative_change_reconstruction_error = (abs(reconstruction_error[-2]-reconstruction_error[-1]))/reconstruction_error[-2] # chance that maybe increases slightly

        iterations +=1

        if verbose:
            print(f'Epoch: {iterations}; reconstruction error: {error}; reconstruction error change: {relative_change_reconstruction_error}')

        epochs += 1
        epochs_plot.append([epochs,error])
    print(pd.DataFrame(epochs_plot, columns=['epoch','reconstruction error']))
    plot = ggplot(pd.DataFrame(epochs_plot, columns=['epoch','reconstruction error']),aes(x='epoch',y='reconstruction error')) + geom_point() + geom_line() 
    print(plot)
    plt.clf()

    return A,B,D,error