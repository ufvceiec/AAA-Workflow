
# LIBRERIAS

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
import itertools
import pydot
import graphviz

from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from IPython.display import display

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy.stats import chi2_contingency, ttest_ind, fisher_exact
from scipy.stats import kruskal
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, TomekLinks, EditedNearestNeighbours, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.combine import SMOTETomek, SMOTEENN

from missingpy import MissForest

import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000)

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import copy

import category_encoders as ce
import os

import shap
shap.initjs()

from sklearn import tree, ensemble, metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus




# WORKFLOW


# PREPROCESAMIENTO DE DATOS

    
"""
   Carga, limpieza y eliminación manual de columnas del dataset.
"""
def load_and_clean(path):
    
    df = dataset_load(path)
    df = bools_and_lower(df)
    
    return df
    
    

"""
   Carga y estandarizacion(valores nan) del dataset en un DataFrame.
"""
def dataset_load(path):
    
    # Lista con posibles tipos de missing values que podemos encontrar en el dataset original.
    missing_values = [' ', 'NaN', 'na', 'Na', '-', '--', 'n/a']
    df = pd.read_excel(path, na_values = missing_values)
    print(df.shape)
    
    return df


"""
   Estandarizacion de los datos.
"""
def bools_and_lower(df):
    
    # Cogemos todas las columnas del DataFrame
    df_cols = df.select_dtypes(include=[np.object]).columns
    # Pasamos a minusculas todos los elementos de cada columna.
    df[df_cols] = df[df_cols].apply(lambda x: x.str.lower())

    # Estandarizamos los valores bool de 'si' y 'no' ya que aparecen elementos con acentos y otros no.
    df = df.replace(['Sí', 'sí', 'SI', 'Si'], 'si')
    df = df.replace(['No', 'NO'], 'no')
    
    return df


"""
   Eliminacion manual de aquellas columnas que consideramos que no son relevantes.
"""
def drop_initials_columns(df):
    
    # Guardamos manualmente las columnas que queremos eliminar.
    cols_to_drop = np.array([
                                        'ID',
                                        'FECHAINTQX',
                                        'NACIONALIDAD',
                                        'FECHADEALTAHOSPITALARIA',
                                        'FECHAREINTERVENCIÓN',
                                        'FECHAEXITUS', 
                                        'CASO',
                                        'COMPL30D',
                                        'NOO_TTOMED',
                                        'NOO_TECNICARECURSOCHIMANCHORSCUFFSEMBOLIZAIBE',
                                        'NOO_ANATOMIAAOAOILIACOILIACO',
                                        'NOO_COMPL30DCARDIORENALNEURONEUMO',
                                        'NOO_CAUSADELAMUERTEA30D',
                                        'NOO_Otra_compli_post2',
                                        'OTRASOBSERVACIONES',
                                        'Otros',
                                        'Exitus_seg',
                                        'F_exitus_seg',
                                        'Exitus_aneurisma_seg',
                                        'Otro_motivo_exitus_seg', 
                                        'Reinter_aneurisma_seg', 
                                        'Fecha_reinter_ane_seg', 
                                        'Presencia_Endoleak_12seg', 
                                        'Diam_max_12_seg',
                                        'Presencia_Endoleak_24seg',
                                        'Diam_max_24_seg'])

    # Eliminacion de las columnas en el DataFrame.
    df_filtrado = df.drop(columns=cols_to_drop)

    return df_filtrado



# TESTS DE CORRELACIÓN


"""
   Test chi cuadrado de independencia de las variables respecto a la variables objetivo. No fue usado finalmente en el proyecto debido a la naturaleza de los datos.
"""
def chi_squared_test(df, target, bias=0.05):
    
    col_escogidas = []
    col_drop_chi2 = []
    n = df.shape[0]

    for i in df.columns:
        tabla_cont = pd.crosstab(df[target], df[i], margins=False).values
        chi2, p_valor, degrees_of_freedom, f_esp = chi2_contingency(tabla_cont)
        col_drop_chi2.append(i)

        if(p_valor<=bias):
            col_escogidas.append(i)
            col_drop_chi2.remove(i)
    
    df = df.drop(columns=col_drop_chi2)

    return df


"""
   Test de correlación de Kruskal-Wallis de las variables respecto a la variables objetivo.
"""
def kruskal_wallis_test(df, target, bias=0.05, plot=False):
    
    col_drop_kruskal = []
    col_escogidas = []
    n = df.shape[0]

    for i in df.columns:
        stat, p_valor = kruskal(df[target], df[i], nan_policy = 'omit')
#         Guardamos en la lista todas las columnas que vamos recorriendo.
        col_drop_kruskal.append(i)
#         print(col_drop_kruskal)
        
        # Comparamos el p-valor obtenido en cada test con el criterio de seleccion que hemos establecido.
        if(p_valor <= bias):
            col_escogidas.append(i)
            col_drop_kruskal.remove(i)
            
            if(plot):
                print('var: {}, p-valor = {}' .format(i, p_valor))
           
    col_drop_kruskal.remove(target)
    df = df.drop(columns=col_drop_kruskal)

    return df



# TECNICAS DE IMPUTACION DE MISSING VALUES



"""
   Relleno de los missing values mediante la moda de cada columna(variable).
"""
def mode_imputation(df, target):

    df_cols_mean = df.columns

    imp_mean = SimpleImputer(strategy='most_frequent')
    imp_mean.fit(df)
    imp_mean_data = imp_mean.transform(df)
    
    data = pd.DataFrame(imp_mean_data, columns=df_cols_mean)
    
    data[target] = data[target].round()
    
    return data



"""
   Relleno de los missing values mediante la media de cada columna(variable).
"""
def mean_imputation(df, target):
    
    df_cols_mean = df.columns

    imp_mean = SimpleImputer(strategy='mean')
    imp_mean.fit(df)
    imp_mean_data = imp_mean.transform(df)
    
    data = pd.DataFrame(imp_mean_data, columns=df_cols_mean)
    
    data[target] = data[target].round()
    
    return data
    

"""
   Relleno de los missing values mediante el algoritmo de knn.
"""
def knn_imputation(df, target):
    
    df_cols_knn = df.columns

    imp_knn_data = fast_knn(df.values, k=5)
    
    data = pd.DataFrame(imp_knn_data, columns=df_cols_knn)
    
    data[target] = data[target].round()
    
    return data


"""
   Relleno de los missing values mediante la técnica de Random Forest.
"""
def random_forest_imputation(df, target):

    
    df_cols_rf = df.columns

    cat_cols = [df.columns.get_loc(col) for col in df.select_dtypes(['float64']).columns.tolist()]
    imp_rf = MissForest()
    imp_rf_data = imp_rf.fit_transform(df, cat_vars=cat_cols)
    
    data = pd.DataFrame(imp_rf_data, columns=df_cols_rf)
    
    data[target] = data[target].round()
    
    return data


# TESTS DE IMPUTACIÓN Y COMPROBACIÓN MISSING VALUES


"""
   Test para el tratamiento de missing values aplicando diferentes técnicas de imputación sobre diferentes porcentajes de missing values permitidos. Tras esto se calcula el MSE para cada una de las combinaciones. El experimento completo se repite un total de 5 veces y se calcula la media debido a la aleatoriedad de algunas de las técnicas de imputación.
"""  
def imputation_tests(df, target):
    
    rf_list = []
    moda_list = []
    knn_list = []
    media_list = []
    
    for porcentaje in [5, 10, 15, 20, 40, 60]:
        
#         print('Porcentaje de missing values: ', porcentaje, '%')
        df_drop = drop_missing_values_columns(df, porcentaje)
#         print('Numero de columnas tras aplicar el porcentaje: ', df_drop.shape[1])
        
        for veces in range(5):
#             print('Prueba-iteracion: ', veces)
            
            datos_rf = random_forest_imputation(df_drop, target)
            mse_rf = mean_squared_error(np.nan_to_num(df_drop), datos_rf, squared=False)

#             print('RMSE = ', mse_rf, 'para imputacion: random forest')
            rf_list.append(mse_rf)

            datos_mode = mode_imputation(df_drop, target)
            mse_moda = mean_squared_error(np.nan_to_num(df_drop), datos_mode, squared=False)
#             print('RMSE = ', mse_moda, 'para imputacion: moda')
            moda_list.append(mse_moda)

            datos_knn = knn_imputation(df_drop, target)
            mse_knn = mean_squared_error(np.nan_to_num(df_drop), datos_knn, squared=False)
#             print('RMSE = ', mse_knn, 'para imputacion: knn')
            knn_list.append(mse_knn)

            datos_mean = mean_imputation(df_drop, target)
            mse_media = mean_squared_error(np.nan_to_num(df_drop), datos_mean, squared=False)
#             print('RMSE = ', mse_media, 'para imputacion: media')
            media_list.append(mse_media)

            print()
        
        print()
        print('---------------------------------------------------------')
        print('Porcentaje de missing values: ', porcentaje, '%')
        print('Media de las 5 iteraciones')
        rf_mean = np.mean(rf_list)
        print('RMSE = ', mse_rf, 'para: random forest')

        moda_mean = np.mean(moda_list)
        print('RMSE = ', mse_moda, 'para: moda')
        
        knn_mean = np.mean(knn_list)
        print('RMSE = ', mse_knn, 'para: knn')
        
        mean_mean = np.mean(media_list)
        print('RMSE = ', mse_media, 'para: media')
        print('---------------------------------------------------------')
        print()


"""
   Eliminacion de aquellas columnas que no cumplen con un porcentaje máximo de missing values.
"""
def drop_missing_values_columns(df, porcentaje):
    
    # A partir del porcentaje, calculamos cuantos registros totales como maximo de missing values puede haber en cada columna.
    null_permitidos = round((len(df)*porcentaje)/100)

    null_columns = []

    for i in df.columns:
        # Contamos el total de missing values de cada columna y si es mayor al total establecido.
        if(df[i].isna().sum() > null_permitidos):
            # Guardamos la columna en la lista.
            null_columns.append(df[i].name)

    # Eliminamos las columnas del DataFrame
    df_filtrado = df.drop(columns=null_columns)
    
    return df_filtrado



"""
   Comprobación de missing values en el DataFrame.
"""
def check_empty_nan(df):
    
    # Para cada fila del DataFrame
    for index, row in df.iterrows():
        # Comprobamos los valores vacios que puede haber.
        empty_values = (row==' ').sum()
        # Comprobamos si quedan NaN.
        nan_values = row.isna().sum()
        # Sumamos los resultado obtenidos.
        total = nan_values+empty_values
        null_permitidos = round((df.shape[1]*total)/100)
        print('- fila: {}, con {} missing values \n' .format(index, null_permitidos))




# TÉCNICAS DE CODIFICACION DE DATOS


"""
   Codificacion del dataset mediante One Hot Encoding.
"""
def one_hot_encoding(df): 
    
    # Seleccionamos las columnas que queremos codificar, en este caso todas las categoricas.
    cat_columns = df.select_dtypes(exclude=['float64', 'int64', 'bool']).columns
    # Aplicamos la codificacion a las columnas seleccionadas.
    df = pd.get_dummies(df, prefix=cat_columns)
        
    return df



"""
   Codificacion de las categorias del dataset a valores numericos.
"""
def ordinal_encoding(df):

    df_temp = df.astype("str").apply(preprocessing.LabelEncoder().fit_transform)
    data = df_temp.where(~df.isna(), df)
    
    data = data.astype('float64')
        
    return data



"""
   Codificacion de las categorias del dataset a valores binarios.
"""
def binary_encoding(df):

    be = ce.BinaryEncoder(return_df=True)
    df = be.fit_transform(df)
        
    return df




# TÉCNICAS HÍBRIDAS (oversampling primero y undersampling después) PARA EL BALANCEO DE CATEGORÍAS


"""
    SMOTE + Tomed Links
"""
def smote_tomed_link(df, target):

    y = df[target]
    X = df.drop(columns=[target])
    
    smote_tl = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority', random_state=42))
    X_resampled, y_resampled = smote_tl.fit_sample(X, y)
    # Redimensionamos de las clases.
    y_resampled = y_resampled.reshape(np.size(X_resampled, 0),1)
    
    cols_df = df.columns
    cols_df = cols_df.drop(target)
    
    data = pd.DataFrame(X_resampled, columns=cols_df)
    data[target] = y_resampled
                
    return data



"""
    SMOTE + Edited Nearest Neighbors
"""
def smote_edited_nearest_neighbor(df, target):

    y = df[target]
    X = df.drop(columns=[target])
    
    smote_enn = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority', random_state=42))
    X_resampled, y_resampled = smote_enn.fit_sample(X, y)
    # Redimensionamos las clases.
    y_resampled = y_resampled.reshape(np.size(X_resampled, 0),1)
    
    cols_df = df.columns
    cols_df = cols_df.drop(target)
    
    data = pd.DataFrame(X_resampled, columns=cols_df)
    data[target] = y_resampled
                
    return data



# TÉCNICAS DE NORMALIZACION Y ESTANDARIZACIÓN


"""
   Normalización minmax de los datos: y = (x-min)/(max-min)
"""
def min_max_normalization(df):
    
    # Cogemos las columnas númericas del dataset para aplicarles la normalización.
    cat_columns_normalize = df.select_dtypes(include=['float64', 'int64']).columns
    min_max_scaler = preprocessing.MinMaxScaler()

    # Para cada columna
    for i in cat_columns_normalize:

        # Cogemos los valores
        x = df[[i]].values
        # Aplicamos la normalización
        x_scaled = min_max_scaler.fit_transform(x)
        # Guaradmos en la columna los nuevos valores.
        df[i] = x_scaled
    
    return df



"""
   Estandarización de los datos: z = (x - u) / s
"""
def standard_scaler(df):
    
    # Cogemos las columnas númericas del dataset para aplicarles la normalización.
    cat_columns_scaler = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = preprocessing.StandardScaler()

    # Para cada columna
    for i in cat_columns_scaler:

        # Cogemos los valores
        x = df[[i]].values
        # Aplicamos la estandarización
        x_scaled = scaler.fit_transform(x)
        # Guaradmos en la columna los nuevos valores.
        df[i] = x_scaled
    
    return df



# PREPARACIÓN DATOS Y CREACIÓN MODELOS NEURONALES (MLP)


"""
   Preparación de los datos para pasarselos a los algoritmos de clasificación y al modelo neuronal, creación conjuntos train y test.
"""
def prep_datos_red(target, df, df_encoding):
        
    df_target = df[target]
    
    values = df_target.unique()
    
    network_output = df_target.map({values[0]:0, values[1]:1})
    network_output = np.asarray(network_output)
    network_output = network_output.reshape(np.size(df, 0),1)
    
    df = df.drop(columns=[target])
    cols_df = df.columns
    
    x_train, x_test, y_train, y_test = train_test_split(df_encoding, network_output, test_size=0.2, stratify=network_output, random_state=42, shuffle=True)
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    
    x_train = x_train.values
    x_test = x_test.values
    
    return x_train, x_test, y_train, y_test, network_output
    

    
"""
   Creación modelo neuronal secuencial con dos capas ocultas tras las pruebas realizadas mediante GridSearch.
"""
def create_sequential_model(shape=1, activation='relu', dropout_rate=0, neurons=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(shape,)),
        tf.keras.layers.Dense(neurons, activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons, activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])
    
    return model


"""
   Creación modelo neuronal funcional donde podemos variar el número de capas ocultas que queramos tener en el MLP pudiendo hacer pruebas con GridSearch.
"""
def create_functional_model(hidden_layers=1, shape=1, activation='relu', dropout_rate=0, neurons=1):
    
    inputs = tf.keras.Input(shape=(shape,))
    
    hidden_layer = tf.keras.layers.Dense(neurons, activation=activation)(inputs)
    dropout_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)
    
    for deep in range(hidden_layers):
        hidden_layer = tf.keras.layers.Dense(neurons, activation=activation)(dropout_layer)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])
    
    return model


"""
   Técnica de GridSearch donde le pasamos el modelo neeuronal ya construido y realizamos pruebas con la combinación de hiper parámetros que establezcamos y medimos en accuracy de la red. Tras esto se devuelven los parámetros con los que mejores resultados se hayan obtenido.
"""
def grid_search_mlp(x_train, y_train, x_test, y_test):
    start = time.time()
    model = KerasClassifier(build_fn=create_sequential_model, shape=x_train.shape[1], verbose=0)

    batch_size = [16, 32]
    epochs = [50]
    dropout_rate = [0.20, 0.25, 0.30]
    neurons = [4, 8, 12]
#     hidden_layers = [2, 3, 4, 5]
    
    param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate, neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5, iid=True)
    grid_result = grid.fit(x_train, y_train)
    end = time.time()
    tiempo_seg = np.round(end-start)
    tiempo_min = np.round(tiempo_seg/60)
    print('Tiempo total del Grid Search: ', tiempo_min, 'minutos')
    print()
    print("Acurracy de la red: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))
    print()
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f con: %r" % (mean, param))
    
    return grid_result.best_params_

    
"""
   Función para entrenar el modelo neuronal junto a los early stoppings para cortar el entrenamiento antes de que el modelo comience a hacer overfitting.
"""    
def train_model(model, params, x_train, y_train, verbose=2, patience=2, plot=True):

    early_stopping_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience)
    
    early_stopping_acc = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=patience)
    
    history = model.fit(x_train, y_train,validation_split=0.2, epochs=params['epochs'], batch_size=params['batch_size'], verbose=verbose, shuffle=True, callbacks=[early_stopping_loss, early_stopping_acc])
    
    if(plot):
        model.summary()
        plot_history(history)
    
    return model
    
     
"""
   Función a la que le pasamos un modelo neuronal entrenado para realizar prediciones con el y calcular las métricas de validación de resultados (accuracy, precision, recall y F1 Score).
"""    
def predict_model_and_report(model, x_test, y_test, classes, batch_size=32):
    
    predicts = model.predict(x_test)
    predicts = predicts.round()

    print("Classification Report")
    print()
    print(classification_report(y_test,predicts,digits=2))    
    
    cnf_matrix = confusion_matrix(y_test,predicts)
    plot_confusion_matrix(cnf_matrix,classes=classes)
    
    return predicts


"""
   Entrenamiento de multiples MLPs
"""
def train_multiple_models(x_train, y_train, x_test, y_test, n, best_params, path, patience):
    scores = []
    for i in range(n):
        tf.keras.backend.clear_session()
        best_model = create_sequential_model(shape=x_train.shape[1], dropout_rate=best_params['dropout_rate'], neurons=best_params['neurons'])
        best_model_train = train_model(best_model, best_params, x_train, y_train, verbose=0, patience=patience, plot=False)

        predicts = best_model_train.predict(x_test)
        predicts = predicts.round()

        best_model_train.save(path + str(i) + '.h5')
        acc = accuracy_score(y_test, predicts)

        scores.append(acc)

    print(scores)
    media = np.mean(scores)
    print('Media: ', media)
    

    
"""
   carga todos los modelos MLP para realizar el ensemble
"""
# load models from file
def load_all_models(n_models, path, plot=False):
    all_models = []
    for n in range(n_models):
        # define filename for this ensemble
        filename = path + str(n) + '.h5'
        # load model from file
        model = tf.keras.models.load_model(filename)
        # add to list of members
        all_models.append(model)
    
        if(plot):
            print('>loaded %s' % filename)

    return all_models



"""
   Carga de los pesos de varios MLP y se realiza la media de las capas
"""
def model_weight_ensemble(members, weights):
    n_layers = len(members[0].get_weights())
    avg_model_weights = list()

    for layer in range(n_layers):
        layer_weights = np.array([model.get_weights()[layer] for model in members])
        avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
        avg_model_weights.append(avg_layer_weights)

    model = tf.keras.models.clone_model(members[0])
    model.set_weights(avg_model_weights)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
    
    return model



def load_and_ensemble_best_model(n, path, plot=False):
    
    members = load_all_models(n, path, plot)
    n_models = len(members)
    weights = [1/n_models for i in range(1, n_models+1)]
    model = model_weight_ensemble(members, weights)
    
    return model
    
    


def shap_kernel_explainer(old_df_dropped, x_train_values, x_test_values, model):
    
    cols_df = old_df_dropped.columns
    new_df = pd.DataFrame(x_train_values, columns=cols_df)
    new_df.head()
    
    kernel_explainer = shap.KernelExplainer(model.predict, new_df, link="logit")
    kernel_shap_values = kernel_explainer.shap_values(x_test_values, nsamples=100)

    return kernel_explainer, kernel_shap_values, new_df



def shap_deep_explainer(old_df_dropped, x_train_values, x_test_values, model):
    
    cols_df = old_df_dropped.columns
    new_df = pd.DataFrame(x_train_values, columns=cols_df)
    new_df.head()
    
    deep_explainer = shap.DeepExplainer(model, new_df)
    shap_values_deep = deep_explainer.shap_values(x_test_values.values)
    
    return deep_explainer, shap_values_deep, new_df



def shap_prediction_force_plot(explainer, shap_values, new_df):
    
    shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], new_df.iloc[0,:], link="logit")

    


def shap_complete_force_plot(explainer, shap_values, new_df):
    
    shap.force_plot(explainer.expected_value, shap_values[0], new_df, link="logit")
    
    

def shap_summary_plot(shap_values, new_df):

    shap.summary_plot(shap_values, new_df, plot_type='bar')    
    


def features_df(old_df, shap_values, number_features, target):
    
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(old_df.columns, sum(vals))), columns=['feature','feature_importance_value'])
    feature_importance.sort_values(by=['feature_importance_value'], ascending=False,inplace=True)
    feature_importance.head(10)
    
    feature_importance = list(feature_importance.iloc[:number_features]['feature'])
    feature_importance.append(target)
    
    data_balanced_features = old_df[list(feature_importance)].copy()
    data_balanced_features.head()
    
    data_balanced_dropped_features = data_balanced_features.drop(columns=target)
    data_balanced_dropped_features.head()
    
    feature_importance = feature_importance[:-1]

    return data_balanced_features, data_balanced_dropped_features, feature_importance



def cart_decision_tree(feature_df, x_train, y_train, x_test, y_test, criterio):

    feature_cols = list(feature_df.columns)

    cart_tree = tree.DecisionTreeClassifier(max_depth=x_train.shape[1]-1, criterion=criterio)
    cart_tree = cart_tree.fit(x_train, y_train)
    
    predict_model_and_report(cart_tree, x_test, y_test, ['exitus', 'no exitus'])

    rules = tree.export_text(cart_tree, feature_names=feature_cols)
    print(rules)
    
    return cart_tree, rules, feature_cols



def knn_classifier(x_train, y_train, x_test, y_test):

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn = knn.fit(x_train, y_train)
    
    plot_history(knn)
    predict_model_and_report(knn, x_test, y_test, ['exitus', 'no exitus'])



def svm_classifier(x_train, y_train, x_test, y_test):

    svm = SVC(kernel='rbf')
    svm = svm.fit(x_train, y_train)
    
    plot_history(svm)
    predict_model_and_report(svm, x_test, y_test, ['exitus', 'no exitus'])
    
    

def lr_classifier(x_train, y_train, x_test, y_test):

    LR = LogisticRegression()
    LR = LR.fit(x_train, y_train)
    
    plot_history(LR)
    predict_model_and_report(LR, x_test, y_test, ['exitus', 'no exitus'])
    
    

def random_forest_dt(feature_cols, x_train, y_train, x_test, y_test, n_estimators, criterio):

    random_forest = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=x_train.shape[1]-1, criterion=criterio)
    random_forest = random_forest.fit(x_train, y_train)
    random_forest # Extracción del mejor árbol, buscarlo
    
    predict_model_and_report(random_forest.estimators_[0], x_test, y_test, ['no exitus', 'exitus'])
    
    rules = tree.export_text(random_forest.estimators_[0], feature_names=feature_cols)
    print(rules)
    
    return random_forest.estimators_[0], rules


def plot_tree(tree, feature_cols):
    
    dot_data = StringIO()
    
    export_graphviz(tree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols, class_names=['no exitus', 'exitus'])
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('cart_dt_hard.png')
    
    return graph
    
     
def plot_history(history):
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model Accuraccy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    
def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):
    
    title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    