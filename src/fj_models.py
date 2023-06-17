import numpy as np
import matplotlib.pyplot as plt

class BasicFJ:
    def __init__(self, W, x0, lambdas):
        """
        W: NxN numpy array, the adjacency matrix for the network
        x0: Nx1 numpy array, the initial opinion vector
        lambdas: Nx1 numpy array, the stubbornness coefficients for each node
        """

        self.W = W
        self.lambdas = np.diag(lambdas)
        self.n = len(x0)
        self.x = x0

        # add checks here

    def update(self):
        """
        Update the opinion vector using the Basic FJ dynamics.
        """

        next_x = np.linalg.multi_dot(((np.eye(self.n) - self.lambdas), self.W, self.x[:,-1])) + np.dot(self.lambdas, self.x[:,0])
        self.x = np.column_stack((self.x, next_x))

    def simulate(self, T):
        """
        Simulate the model for T time steps.
        """
        for t in range(T):
            self.update()

    def plot(self):
        """
        Plot the opinion vector over time.
        """
        plt.plot(np.transpose(self.x))
        plt.show()

class ControlledFJ:
    def __init__(self,W,x0,lambdas):
        """
        :param W:
        :param x0:
        :param lambdas:
        """
        self.n = len(x0)-1 # number of nodes: -1 because one is recommender
        self.lambdas = np.diag(lambdas[:-1]) # stubbornness coefficients for all nodes except recommender

        # create augmented state vector
        self.x = np.vstack((x0[:-1], x0[:-1]))

        # create A matrix
        self.A = np.block([[np.dot(np.eye(self.n)-self.lambdas,W[:-1,:-1]),self.lambdas],[np.zeros((self.n,self.n)),np.eye(self.n)]])

        # create B matrix
        self.B = np.vstack((W[:-1,-1].reshape(-1,1),np.zeros((self.n,1))))

    def update(self, u):
        """
        Update the opinion vector using the Controlled FJ dynamics.
        """
        next_x = np.dot(self.A,self.x[:,-1].reshape(-1,1)) + np.dot(self.B,u)
        self.x = np.column_stack((self.x, next_x))

def simulate(A,lambdas,x0,K):
    # simulate the uncontrolled model
    x = np.zeros((A.shape[0],K+1))
    x[:,0] = x0
    for k in range(K):
        x[:,k+1] = A @ x[:,k] + lambdas @ x0

    return x
