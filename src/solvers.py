import cvxpy as cp
import numpy as np

def SolveStep(A,B,lambdas,x0,x_init,K):
    # x0 and x_init are not the same. x0 is the initial state of the system before any steps are solved in FJ model. x_init is the initial state of the system after K steps are solved in FJ model.
    nx = x0.shape[0] # number of states
    nu = B.shape[1] # number of inputs

    # initialize variables. x is the state trajectory, u is the input trajectory
    # x is of size K+1 because we need to include the initial state
    x = cp.Variable((nx,K+1))
    u = cp.Variable((nu,K))

    # initialize cost and constraints
    cost = 0
    constr = []

    # add cost and constraints for each timestep
    for k in range(K):
        # multiply u by vector of ones to match size of x
        cost += cp.sum_squares(x[:,k]-u[:,k]*np.ones(nx,))

        # constraint on dynamics and input. Constraint on state is not needed because it is already bounded by the dynamics (?)
        constr += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + lambdas @ x0, u[:,k] <= 1, 0 <= u[:,k]]

    # add initial constraint
    constr += [x[:, 0] == x_init]

    # solve the problem
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()
    # print("Solver status: ",problem.status)
    return x, u, cost

def SolveRecedingHorizon(A,B,lambdas,x0,K,T):
    # initialize empty variables for state trajectory and input trajectory over time horizon T
    nx = x0.shape[0] # number of states
    nu = B.shape[1] # number of inputs

    x = np.zeros((nx,T+1))
    u = np.zeros((nu,T))

    # set initial state
    x[:,0] = x0

    sum_cost = 0
    for t in range(T):
        x_t,u_t,cost_t = SolveStep(A,B,lambdas,x0,x[:,t],K)
        x[:,t+1] = x_t.value[:,1]
        u[:,t] = u_t.value[:,0]
        sum_cost += np.sum((x[:,t]-u[:,t]*np.ones(nx,))**2)

    return x, u, sum_cost

def SolveNaiveController(A,B,lambdas,x0,T):
    # initialize empty variables for state trajectory and input trajectory over time horizon T
    nx = x0.shape[0]  # number of states
    nu = B.shape[1]  # number of inputs
    x = np.zeros((nx,T+1))
    u = np.zeros((nu,T))

    # set initial state
    x[:,0] = x0
    sum_cost = 0

    for t in range(T):
        u_t = cp.Variable((nu,))
        cost = cp.sum_squares(x[:, t] - u_t * np.ones(nx, ))
        constr = [u_t <= 1, 0 <= u_t]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        x[:,t+1] = A @ x[:,t] + B @ u_t.value + lambdas @ x0
        u[:,t] = u_t.value
        sum_cost += cost.value

    return x, u, sum_cost