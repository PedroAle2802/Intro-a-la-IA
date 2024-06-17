from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classifierNaiveBayes(x, y, samp_size=0.2, params=None, accuracy_solicitado=0.9):
    '''
    classifierNaiveBayes(x, y, samp_size=0.2, params=None, accuracy_solicitado=0.9):

    x: matriz de atributos
    y: matriz de datos "target"
    sam_size: valor de tamaño de muestra para prueba del clasificador, por defecto tiene valor 0.2
    params: recibe de entrada un diccionario con los parámetros en interés del clasificador
            y una lista de los valores a probar respectivamente.
            Hay 1 posibilidad de diccionario por defecto por nombre 'param_defecto' donde sus valores están dados por

            param_defecto:
            -var_smoothing : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

    ----------------------------------------------------------------------------------------------------------------------
    accuracy_solicitado: valor de accuracy deseado por el usuario, recibe valores tipo "float" de entre 0 y 1,
                         por defecto tiene por valor 0.9
    ----------------------------------------------------------------------------------------------------------------------
    la salida tiene la siguiente estructura: 

    salida (return):  accuracy_in, accuracy_out, parametros_final, clasificador

            -accuracy_in: Precisión del modelo en el conjunto de entrenamiento.

            -accuracy_out: Precisión del modelo en el conjunto de prueba.

            -parametros_final: Parámetros finales del clasificador.
                        
            -clasificador: Este valor es el propio clasificador. Se devuelve para que principalmente el usuario pueda 
            utilizarlo de manera directa para futuras predicciones adicionales o cualquier otra operación con él.
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=samp_size, random_state=42)
    
    if params is None:
        # Se entrena Naive Bayes por defecto, sin parámetros
        nb = GaussianNB()
    elif params == 'param_defecto':
        # Se realiza Randomized Search para buscar hiperparámetros con el diccionario por defecto
        defecto = {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
        
        random_search = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=defecto, n_iter=10, scoring='accuracy', cv=5, random_state=42)
        
        random_search.fit(X_train, y_train)
        nb = random_search.best_estimator_
    else:
        # Caso contrario: realizar Randomized Search para buscar hiperparámetros
        diccionario_params = params
        random_search = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=diccionario_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)
        random_search.fit(X_train, y_train)
        nb = random_search.best_estimator_

    parametros_final = nb.get_params()
    nb.fit(X_train, y_train)
    # Predicciones
    y_nb_train = nb.predict(X_train)
    y_nb_test = nb.predict(X_test)
    # Métricas
    accuracy_in = accuracy_score(y_train, y_nb_train)
    accuracy_out = accuracy_score(y_test, y_nb_test)
    # Return
    if accuracy_out >= accuracy_solicitado:
        return accuracy_in, accuracy_out, parametros_final, nb
    else:
        print(f'No se ha llegado al accuracy {accuracy_solicitado}, el mejor valor encontrado por medio de Random Search es:')
    
    print(f'accuracy in: {accuracy_in}\n\naccuracy out: {accuracy_out}\n\nparámetros finales: {parametros_final}')
    
    return round(accuracy_in,3), round(accuracy_out,3), parametros_final, nb

