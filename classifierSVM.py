from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


def classifierSVM(x, y, samp_size, params, accuracy_solicitado):
    '''
    classifierSVM(x, y, samp_size=0.2, params=None, accuracy_solicitado=0.9):

    x: matriz de atributos
    y: matriz de datos "target"
    sam_size: valor de tamaño de muestra para prueba del clasificador, por defecto tiene valor 0.2
    params: recibe de entrada un diccionario con los parámetros en interés del clasificador
            y una lista de los valores a probar respectivamente.
            Hay una posibilidad de un diccionario por defecto por nombre 'param_defecto', que su valor está dado por

            -kernel: ['linear', 'rbf', 'poly']
            -C: [0.1, 1, 10, 100]
            -gamma: [0.001, 0.01, 0.1, 1]
            -degree: [2, 3, 4, 5]
    ----------------------------------------------------------------------------------------------------------------------
    accuracy_solicitado: valor de accuracy deseado por el usuario, recibe valores tipo "float" de entre 0 y 1,
                         por defecto tiene por valor 0.9
    ----------------------------------------------------------------------------------------------------------------------
    la salida tiene la siguiente estructura: 

    salida (return):  accuracy_in, acc_out, parametros_final, clasificador

            -accuracy_in: Precisión del modelo en el conjunto de entrenamiento.

            -accuracy_out: Precisión del modelo en el conjunto de prueba.

            -parametros_final: Parámetros finales del clasificador.
            
            -clasificador: Este valor es el propio clasificador. Se devuelve para que, principalmente el usuario pueda 
            utilizarlo de manera directa para futuras predicciones adicionales o cualquier otra operación con él.
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=samp_size, random_state=42)

    if params == None:
        #Entrenar inicialmente con kernel lineal
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        y_svm = svm.predict(X_test)        
        accuracy = accuracy_score(y_true=y_test, y_pred=y_svm)
        
        #Condicional: Si el accuracy es menor al solicitado, intentar con kernel RBF
        if accuracy < accuracy_solicitado:
            svm = SVC(kernel='rbf')
            
    elif params == 'param_defecto':
        # Se realiza Randomized Search para buscar hiperparámetros
        diccionario_params = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]}
        random_search = RandomizedSearchCV(estimator=SVC(), param_distributions=diccionario_params, n_iter=5, scoring='accuracy', cv=2, random_state=42)
        random_search.fit(X_train, y_train)
        svm = random_search.best_estimator_
        y_svm = svm.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_svm)

    else:
        diccionario_params = params
        random_search = RandomizedSearchCV(estimator=SVC(), param_distributions=diccionario_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)
        random_search.fit(X_train, y_train)
        svm = random_search.best_estimator_

    #Desde svm.fit
    parametros_final = svm.get_params()
    svm.fit(X_train, y_train)

    #Predicciones
    y_svm_train = svm.predict(X_train)
    y_svm_test = svm.predict(X_test)

    #Métricas
    accuracy_in = accuracy_score(y_train, y_svm_train)
    accuracy_out = accuracy_score(y_test, y_svm_test)


    #Return
    if (accuracy_out >= accuracy_solicitado):
        return accuracy_in, accuracy_out, parametros_final, svm
    else:
        print(f'No se ha llegado al accuracy {accuracy_solicitado}, el mejor valor es encontrado por medio de Random Search es:')
        return round(accuracy_in, 3), round(accuracy_out, 3), parametros_final, svm