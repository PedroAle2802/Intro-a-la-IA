import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def busqueda_tamano_pca(clasificador, X, y, pca=[], size=[], tamano_random=0.8):
    """
    Esta función realiza una búsqueda aleatoria para determinar la mejor combinación de tamaño de 
    conjunto de prueba y número de componentes principales (PCA) que maximiza la precisión de un clasificador.

    Parámetros:
    - clasificador: El modelo de clasificación a utilizar.
    - X: Matriz de características (features).
    - y: Vector de etiquetas (labels).
    - pca: Lista de posibles valores para el número de componentes principales en PCA.
    - size: Lista de posibles valores para el tamaño del conjunto de prueba.
    - tamano_random: Proporción del total de combinaciones (pca, size) a evaluar aleatoriamente. 
                     Valor entre 0 y 1.

    Retorna:
    - El mejor número de componentes principales (PCA) y el mejor tamaño del conjunto de prueba 
      en términos de precisión del clasificador.
    """
    # Generar todas las combinaciones posibles de valores de pca y size
    lista1 = [[p, s] for p in pca for s in size]
    lista2 = []
    accs = []
    
    # Número de combinaciones a evaluar aleatoriamente
    n = int(tamano_random * len(lista1))
    
    for i in range(n):
        # Seleccionar una combinación aleatoria de pca y size
        par = random.choice(lista1)
        lista1.remove(par)
        # lista2.append(par)
        pca_val, size_val = par[0], par[1]
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=size_val,
                                                            random_state=42)
        
        # Crear un pipeline con PCA y el clasificador
        pipeline = make_pipeline(PCA(n_components=pca_val),
                                 clasificador)
        
        # Entrenar el pipeline con los datos de entrenamiento
        pipeline.fit(X_train, y_train)
        
        # Predecir las etiquetas para los datos de prueba
        y_pred = pipeline.predict(X_test)
        
        # Calcular la precisión del pipeline en los datos de prueba
        accuracy = pipeline.score(X_test, y_test)
        lista2.append([pca_val, size_val,accuracy])
        accs.append(accuracy)
    
    # Encontrar la máxima precisión obtenida
    max_acc = max(accs)
    
    # Buscar la combinación (pca, size) correspondiente a la máxima precisión
    for i in range(n):
        if accs[i] == max_acc:
            par_max = lista2[i]
            break
    # Extraer datos para la gráfica
    X_vals = np.array([par[0] for par in lista2])
    Y_vals = np.array([par[1] for par in lista2])
    Z_vals = np.array([par[2] for par in lista2])
    
    # Crear la malla de datos
    X_unique = np.unique(X_vals)
    Y_unique = np.unique(Y_vals)
    X_mesh, Y_mesh = np.meshgrid(X_unique, Y_unique)
    Z_mesh = np.zeros(X_mesh.shape)
    
    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]
            mask = (X_vals == x_val) & (Y_vals == y_val)
            if np.any(mask):
                Z_mesh[i, j] = Z_vals[mask][0]
    
    # Crear la figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar la superficie
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis')
    # fig.colorbar(surf)
    
    # Etiquetas de los ejes
    ax.set_xlabel('PCA')
    ax.set_ylabel('Tamaño')
    ax.set_zlabel('Precisión')
    
    # Título
    ax.set_title('Accuracy en función del\nnúmero decomponentes PCA y\ntamaño del conjunto de prueba')
    # Anotar el punto de máxima precisión
    max_pca, max_size, max_accuracy = par_max
    ax.scatter(max_pca, max_size, max_accuracy, color='red', s=100)
    ax.text(max_pca, max_size, max_accuracy, f'  (pca: {max_pca},tamaño:{max_size},acc = {max_accuracy:.2f})', color='red')
    
    # Mostrar la gráfica
    plt.show()
    
    
    return max_pca , max_size , par_max[2]


# from sklearn.ensemble import RandomForestClassifier

# clasificador = RandomForestClassifier()
# X = np.random.rand(100, 10)
# y = np.random.randint(2, size=100)
# pca_vals = [2, 3, 4, 5,6]
# size_vals = [0.2, 0.3, 0.4, 0.5,0.6,.7]

# mejor_pca, mejor_size = busqueda_tamano_pca(clasificador, X, y, pca=pca_vals, size=size_vals,tamano_random=1)
# print(f"Mejor número de componentes PCA: {mejor_pca}")
# print(f"Mejor tamaño del conjunto de prueba: {mejor_size}")

# from classifierLR import LogisticRegression
# from sklearn.svm import SVC
# from busqueda_acc_pca_tamaño import busqueda_tamano_pca
# import pandas as pd
# from sklearn.datasets import load_digits
# from sklearn.tree import  DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier()
# tree = DecisionTreeClassifier()
# digits = load_digits()
# X = digits.data  # Características (imágenes de dígitos)
# y = digits.target                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             # Etiquetas (dígitos de 0 a 9)
# # print(digits.DESCR)

# # help(classifierLogisticRegression)
# clf = LogisticRegression()#classifierLogisticRegression(X, y, samp_size=0.2, params=None, accuracy_solicitado=0.9)
# df = pd.read_csv('diabetes.csv')
# # X = df[df.columns[0:-1]]
# # y = df[df.columns[-1]]
# svm = SVC()
# mejor_pca, mejor_size = busqueda_tamano_pca(svm, X, y,
#                                              pca=[i for i in range(1,20)],
#                                              size=[i/20 for i in range(4,20)],
#                                              tamano_random=1)
# print(f"Mejor número de componentes PCA: {mejor_pca}")
# print(f"Mejor tamaño del conjunto de prueba: {mejor_size}")
# # print(digits.images[0])
