import numpy as np
import math


class ADMMSolver:
    '''ADMM Solver for convex optimisation tasks'''
    def __init__(self, lamb, num_stacked, size_blocks, rho, S,
                 rho_update_func=None):
        self.lamb = lamb
        self.numBlocks = num_stacked
        self.sizeBlocks = size_blocks
        probSize = num_stacked*size_blocks
        self.length = int(probSize*(probSize+1)/2)
        self.x = np.zeros(self.length)
        self.z = np.zeros(self.length)
        self.u = np.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = 'initialized'
        self.rho_update_func = rho_update_func

    def ij2symmetric(self, i, j, size):
        return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i

    def upper_to_full(self, a):
        n = int((-1 + np.sqrt(1 + 8*a.shape[0]))/2)
        A = np.zeros([n, n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)
        return A

    def Prox_logdet(self, S, A, eta):
        d, q = np.linalg.eigh(eta*A-S)
        X_var = (1/(2*float(eta)))*q@(np.diag(d + np.sqrt(
            np.square(d) + (4*eta)*np.ones(d.shape))))@q.T
        # extract upper triangular part as update variable
        x_var = X_var[np.triu_indices(S.shape[1])]
        return x_var.T

    def ADMM_x(self):
        a = self.z-self.u
        A = self.upper_to_full(a)
        eta = self.rho
        x_update = self.Prox_logdet(self.S, A, eta)
        self.x = np.array(x_update).T.reshape(-1)

    def ADMM_z(self, index_penalty=1):
        """Consensus variable Z - a block Toeplitz matrix"""
        a = self.x + self.u
        probSize = self.numBlocks*self.sizeBlocks
        z_update = np.zeros(self.length)

        # TODO: can we parallelize these?
        # i=0 case
        elems = self.numBlocks  # i=0 is diagonal
        for j in range(self.sizeBlocks):
            startPoint = j
            for k in range(startPoint, self.sizeBlocks):
                locList = [(l*self.sizeBlocks + j, l*self.sizeBlocks+k) for l in range(int(elems))]
                lamSum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in locList)
                indices = [self.ij2symmetric(loc1, loc2, probSize) for (loc1, loc2) in locList]
                pointSum = sum(a[int(index)] for index in indices)
                rhoPointSum = self.rho * pointSum

                # Calculate soft threshold
                ans = 0
                # If answer is positive
                if rhoPointSum > lamSum:
                    ans = max((rhoPointSum - lamSum)/(self.rho*elems), 0)
                elif rhoPointSum < -1*lamSum:
                    ans = min((rhoPointSum + lamSum)/(self.rho*elems), 0)

                for index in indices:
                    z_update[int(index)] = ans

        #  case i > 0
        for i in range(1, self.numBlocks):
            elems = (2*self.numBlocks - 2*i)/2
            for j in range(self.sizeBlocks):
                startPoint = 0
                for k in range(startPoint, self.sizeBlocks):
                    locList = [((l+i)*self.sizeBlocks + j, l*self.sizeBlocks+k) for l in range(int(elems))]
                    lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
                    indices = [self.ij2symmetric(loc2, loc1, probSize) for (loc1, loc2) in locList]
                    pointSum = sum(a[int(index)] for index in indices)
                    rhoPointSum = self.rho * pointSum

                    # Calculate soft threshold
                    ans = 0
                    # If answer is positive
                    if rhoPointSum > lamSum:
                        ans = max((rhoPointSum - lamSum)/(self.rho*elems), 0)
                    elif rhoPointSum < -1*lamSum:
                        ans = min((rhoPointSum + lamSum)/(self.rho*elems), 0)

                    for index in indices:
                        z_update[int(index)] = ans
        self.z = z_update

    def ADMM_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = x - z
    # s = rho * (z - z_old)
    # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
    # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def CheckConvergence(self, z_old, e_abs, e_rel):
        norm = np.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    # solve
    def __call__(self, maxIters, eps_abs, eps_rel):
        self.status = 'Incomplete: max iterations reached'
        z_old = np.copy(self.z)
        self.ADMM_x()
        self.ADMM_z()
        self.ADMM_u()

        for i in range(1, maxIters):
            z_old = np.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel)
            if stop:
                self.status = 'Optimal'
                break
            new_rho = self.rho
            if self.rho_update_func:
                new_rho = self.rho_update_func(self.rho, res_pri, e_pri,
                                               res_dual, e_dual)
            scale = self.rho / new_rho
            rho = new_rho
            self.u = scale*self.u

        return self.upper_to_full(self.x)
