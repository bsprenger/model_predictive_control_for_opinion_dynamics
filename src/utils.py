import numpy as np

def generate_random_W_and_B(nx,nu,density=1,control_influence=1):
    matrix = np.zeros((nx+nu,nx+nu))

    for i in range(nx+nu):
        for j in range(nx+nu):
            if np.random.rand() < density:
                matrix[i,j] = np.random.rand()

    matrix[:nx,nx:] = control_influence * matrix[:nx,nx:]
    matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
    W = matrix[:nx,:nx]
    B = matrix[:nx,nx:]
    return W,B

def generate_random_dynamics(nx,nu,bias_low = 0,bias_high = 1,density = 1,control_influence = 1,stubborn_nodes = 0):
    # W,B = generate_random_W_and_B(nx,nu,density,control_influence)

    matrix = np.zeros((nx + nu, nx + nu))
    for i in range(nx+nu):
        for j in range(nx+nu):
            if np.random.rand() < density:
                matrix[i,j] = np.random.rand()

    matrix[:, nx:] = control_influence * matrix[:, nx:]
    matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]

    x0 = np.random.rand(nx,)
    lambdas = np.diag(np.random.uniform(bias_low, bias_high, nx + 1))

    if stubborn_nodes > 0:
        stubborn_nodes = np.random.choice(nx, stubborn_nodes, replace=False)
        lambdas[stubborn_nodes, stubborn_nodes] = 1.0

    matrix = (np.eye(nx+1) - lambdas) @ matrix

    A = matrix[:nx,:nx]
    B = matrix[:nx,nx:]
    lambdas = lambdas[:-1,:-1]

    return A, B, lambdas, x0

def check_valid_dynamics(A,B,lambdas):
    nx = A.shape[0]
    nu = B.shape[1]
    assert A.shape == (nx,nx)
    assert B.shape == (nx,nu)
    assert lambdas.shape == (nx,nx)
    assert np.all(np.linalg.eigvals(A) < 1)
    assert np.all(np.linalg.eigvals(lambdas) < 1)
    assert np.all(np.linalg.eigvals(A - np.dot(lambdas,W)) < 1)


