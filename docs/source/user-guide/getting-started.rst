import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
pi = np.pi
nx, ny = 6, 6  # Número de pontos na grade (pi/h + 1, pi/(2k) + 1)
lx, ly = pi, pi/2  # Tamanho do domínio
h, k = pi / 5, pi / 10  # Espaçamento da grade

# Criação da grade
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
u = np.zeros((nx, ny))

# Condições de contorno
u[0, :] = np.cos(y)  # Borda esquerda (x=0)
u[-1, :] = -np.cos(y)  # Borda direita (x=pi)
u[:, 0] = np.cos(x)  # Borda inferior (y=0)
u[:, -1] = 0  # Borda superior (y=pi/2)

# Parâmetros de iteração
tolerance = 1e-5
max_iter = 10000

def solve_laplace(u, h, k, tolerance, max_iter):
    for iteration in range(max_iter):
        u_new = u.copy()
        # Atualização dos pontos internos
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_new[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] 
                                      - h**2 * (-np.cos(x[i] + y[j]) - np.cos(x[i] - y[j])))
        
        # Verificação da convergência
        if np.max(np.abs(u_new - u)) < tolerance:
            print(f'Convergência alcançada após {iteration + 1} iterações')
            break
        
        u = u_new

    return u

# Resolvendo a equação de Laplace
u = solve_laplace(u, h, k, tolerance, max_iter)

# Plotando o resultado
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u.T, 20, cmap='viridis')
plt.colorbar(label='$u(x,y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solução da Equação de Laplace')
plt.show()
