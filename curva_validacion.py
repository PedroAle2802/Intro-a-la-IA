import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.preprocessing import StandardScaler
from busqueda_acc_pca_tamaño import busqueda_tamano_pca
from sklearn.pipeline import make_pipeldine
def curva_aprendizaje(clasificador)