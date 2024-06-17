import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def grafica_PCA(X):
    # Escalar los datos
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # Calcular la matriz de covarianza
    cov_mat = np.cov(X_std.T)
    
    # Calcular valores y vectores propios
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    # Calcular la varianza explicada
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    # Visualizar la varianza explicada
    num_components = len(var_exp)
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, num_components + 1), var_exp, alpha=0.5, align='center', label='Varianza explicada individual')
    plt.step(range(1, num_components + 1), cum_var_exp, where='mid', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de varianza explicada')
    plt.xlabel('Componentes principales')
    plt.legend(loc='best')
    plt.title('PCA: Varianza explicada por componentes principales')
    plt.show()
    return 0

