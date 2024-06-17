
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def classifierTree(x,y,size,params):
    #Se dividen los datos en Entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
    if params == None:
        #Se instancia el clasificador
        clf = DecisionTreeClassifier()

        #Se entrena el clasificador
        clf.fit(X_train, y_train)

        #Se obtienen las predicciones
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        #print(f"La accuracy es: {accuracy:.2f}")
        return accuracy_in,accuracy_out,clf.get_params(),clf
    
    elif params == 'param_defecto':
        # Define el espacio de búsqueda de hiperparámetros
        param_dist = {
            'criterion': ['gini', 'entropy'],  # Tipo de criterio
            'splitter': ['best', 'random'],    # Estrategia de división
            'max_depth': np.arange(1, 11),    # Profundidad máxima del árbol
            'min_samples_split': np.arange(2, 11),  # Mínimo de muestras para dividir un nodo
            'min_samples_leaf': np.arange(1, 11)  # Mínimo de muestras en una hoja
            #'ccp_alpha': list(np.linspace(0.0, 0.2, 100))
        }

        # Crea un clasificador de árbol de decisión
        clf = DecisionTreeClassifier(random_state=42)
        #print(clf.get_params())
        
        # Crea un objeto RandomizedSearchCV
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, scoring='accuracy')

        clf = random_search
        # Realiza la búsqueda aleatoria en los datos de entrenamiento
        clf.fit(X_train, y_train)
        
        clf = clf.best_estimator_
        #Se obtienen las predicciones
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        #print(f"La accuracy es: {accuracy:.2f}")
        return accuracy_in,accuracy_out,clf.get_params(),clf
    
    else:
        diccionario_params = params
        # Crea un clasificador de árbol de decisión
        clf = DecisionTreeClassifier(random_state=42)
        print(clf.get_params())
        # Crea un objeto RandomizedSearchCV
        random_search = RandomizedSearchCV(clf, param_distributions=diccionario_params, n_iter=100, cv=5, random_state=42, scoring='accuracy')

        # Realiza la búsqueda aleatoria en los datos de entrenamiento
        random_search.fit(X_train, y_train)
        
    
        #Se obtienen las predicciones
        y_pred_train = random_search.predict(X_train)
        y_pred_test = random_search.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        #print(f"La accuracy es: {accuracy:.2f}")
        return round(accuracy_in,3), round(accuracy_out,3),random_search.best_estimator_.get_params(),random_search.best_estimator_