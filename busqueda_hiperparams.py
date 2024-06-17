from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

## 1. Regresión Logística (Logistic Regression)
# solver: Algoritmo a usar en el problema de optimización (e.g., 'liblinear', 'lbfgs', 'sag', 'saga').
# C: Parámetro de regularización inversa.
# penalty: Tipo de penalización ('l1', 'l2', 'elasticnet', 'none').
# max_iter: Número máximo de iteraciones.
## 2. Máquina de Vectores de Soporte (Support Vector Machine - SVM)
# C: Parámetro de regularización.
# kernel: Tipo de kernel a usar ('linear', 'poly', 'rbf', 'sigmoid').
# gamma: Coeficiente para los kernels 'rbf', 'poly' y 'sigmoid'.
# degree: Grado del polinomio (si kernel='poly').
# coef0: Término independiente en el kernel de polinomio y sigmoide.
## 3. Árboles de Decisión (Decision Tree)
# criterion: Función para medir la calidad de una división ('gini', 'entropy').
# splitter: Estrategia usada para elegir la división ('best', 'random').
# max_depth: Profundidad máxima del árbol.
# min_samples_split: Número mínimo de muestras necesarias para dividir un nodo.
# min_samples_leaf: Número mínimo de muestras necesarias en un nodo hoja.
## 4. Bosques Aleatorios (Random Forest)
# n_estimators: Número de árboles en el bosque.
# criterion: Función para medir la calidad de una división ('gini', 'entropy').
# max_depth: Profundidad máxima del árbol.
# min_samples_split: Número mínimo de muestras necesarias para dividir un nodo.
# min_samples_leaf: Número mínimo de muestras necesarias en un nodo hoja.
# max_features: Número de características a considerar al buscar la mejor división.
## 5. Naive Bayes
# var_smoothing: Porción de la mayor varianza de los componentes, agregada a la varianza para la estabilidad numérica.


def busqueda_hiperparams(clf, param_grid, X, y, method='grid', cv=5, scoring='accuracy', n_iter=100):
    """
    Realiza una búsqueda de hiperparámetros utilizando GridSearchCV o RandomizedSearchCV.
    
    Parámetros:
    - clf: Clasificador (e.g., RandomForestClassifier())
    - param_grid: Diccionario con los hiperparámetros a buscar
    - X: Características del conjunto de datos
    - y: Etiquetas del conjunto de datos
    - method: Método de búsqueda ('grid' para GridSearchCV, 'random' para RandomizedSearchCV)
    - cv: Número de pliegues en la validación cruzada (default: 5)
    - scoring: Métrica para evaluar el rendimiento (default: 'accuracy')
    - n_iter: Número de iteraciones para RandomizedSearchCV (default: 100)
    
    Retorna:
    - best_model: Modelo con los mejores hiperparámetros
    - best_params: Los mejores hiperparámetros encontrados
    """
    if method == 'grid':
        search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring=scoring)
    elif method == 'random':
        search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=n_iter, cv=cv, scoring=scoring, random_state=42)
    else:
        raise ValueError("El método debe ser 'grid' o 'random'")

    search.fit(X, y)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    return best_model, best_params