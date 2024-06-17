import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def varianza_explicada_acumulada(X):
    # Estandarizar los datos
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X)

    # Calcular la matriz de covarianza
    cov_mat = np.cov(X_train_std.T)

    # Calcular valores y vectores propios
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('\nEigenvalues \n%s' % eigen_vals)

    # Calcular la varianza explicada
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # Graficar la varianza explicada
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center', label='Varianza explicada individual')
    plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de varianza explicada')
    plt.xlabel('Componentes principales')
    plt.title('Varianza explicada por componentes principales')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return 0

def varianza_explicada_c_comp(X):
    pca = PCA()
    pca.fit(X)

    # Obtener la varianza explicada por cada componente principal
    explained_variance_ratio = pca.explained_variance_ratio_

    # Graficar la varianza explicada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.title('Varianza Explicada por Componentes Principales')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid()
    plt.show()
    return 0

def varianza_acumulada(X):
    pca = PCA()
    pca.fit(X)
    # Obtener la varianza explicada por cada componente principal
    explained_variance_ratio = pca.explained_variance_ratio_
    # Graficar la varianza acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
    plt.title('Varianza Explicada Acumulada por Componentes Principales')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid()
    plt.show()
    return 0