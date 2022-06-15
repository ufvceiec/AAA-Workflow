# Workflow para la predicción de éxito mediante modelos DeepLearning y la posterior creación de un sistema basado en reglas

Este proyecto consiste en la creación de un sistema basado en reglas a partir de la predicción de éxito de operaciones de AAA. Para su desarrollo han sido aplicadas un gran conjunto de técnicas de Machine Learning y Deep Learning. Está pensando para el trabajo con conjuntos de datos de clasificación binaria en los que una de las categorías de la clase a predecir se encuentra desequilibrada o desbalanceada. Para ello contamos con técnicas de oversamplingg y undersampling en el workflow para tratar este problema.


## Organización del proyecto

A continuación se describe la organización de las diferentes carpetas del proyecto:

- En [instalación_librerías](/instalación_librerías) - podemos encontrar el archivo ¨*requirements.txt*¨ que podemos instalar en un enviroment nuevo para instalar todas las librerías que han sido usadas en el proyecto.

- En [datasets_generados](/datasets_generados) - tenemos los datos originales usados en el proyecto junto a las diferentes versiones que se han ido creando tras aplicar las diferentes técnicas descritas en el proyecto.

- En [models](/models) - tenemos los mejores modelos neuronales que han sido entrenados para la predicción de éxito de un AAA, tanto para el conjunto de datos balanceado en la carpeta [normal](/models/normal), como para el conjunto de datos tras aplicar las técnicas de feature engineering en [features](/models/features).

- En [pruebas_datasets](/pruebas_datasets) - tenemos los datasets utilizados para el desarollo de la segunda prueba de concepto establecida en el proyecto, junto a los modelos neuronales generados tras la aplicación de las diferentes técnicas utilizadas.

- En [SurrogateTrees](/SurrogateTrees) - tenemos los scripts de código que han sido utilizados para el uso del algoritmo TREPAN descrito en la memoria del proyecto. Este código ha sido extraido de una librería más grande conocida como [skater](https://github.com/oracle/Skater) y modificado para su uso en el proyecto.

- En el resto del directorio podemos encontrar todos los notebooks de jupyter que han sido creados. También podemos encontrar el script de código principal, workflow.py donde están implementadas todas las técnicas y funciones que se han utilizado:

  - En ¨*grid_search_datos_originales.ipynb*¨ tenemos realizado un entrenamiento de un modelo neuronal con los datos originales, como se puede ver, debido al desequilibrio en la distribución de las clases de la varible a predecir, el modelo no es capaz de aprender correctamente
  - En ¨*Balanceo_smote_enn.ipynb*¨ se ha aplicado la técnica de muestreo híbrida formada por SMOTE y ENN y se han realizado pruebas entrenando modelos neuronales para comprobar su correcto funcionamiento
  - En ¨*Balanceo_smote_tomed_link.ipynb*¨ se ha aplicado la técnica de muestreo híbrida formada por SMOTE y Tomed Links y se han realizado pruebas entrenando modelos neuronales para comprobar su correcto funcionamiento
  - En ¨*pruebas_datasets_desbalancedos.ipynb*¨ podemos encontrar las pruebas realizadas sobre los datasets de cáncer de mama y derrames de petroleo, los cuales poseen una distribución de clases desequilibrada. El objetivo es probar todo el workflow y las técnicas de balanceo con conjuntos de datos diferentes al original. Primero se han entrenado modelos con los datos originales, a continuación se han aplicado las técnicas de muestreo y por último se han vuelto a entrenar modelos con los datos desbalanceados, para así comprobar el correcto funcionamiento de los modelos neuronales tras aplicar las técnicas de muestreo.
  - En ¨*shap_prueba_features.ipynb*¨ tenemos pruebas realizadas sobre la extracción de las características más influyentes a partir de un modelo neuronal entrenado. Podemos encontrar pruebas con dos aproximaciones: *Kernel Explainer* y *Deep Explainer*.
  - En ¨*flujo_completo.ipynb*¨ encontramos el flujo completo, desde la carga, preprocesamiento y desbalanceo de los datos, hasta el entrenamiento de modelos neuronales, la extración de las características más influyentes en las decisiones de los mismos y la generación del conjunto de reglas por medio de distintas aproximaciones basadas en Árboles de Decisión.
  - En ¨*workflow.py*¨ podemos encontrar definidas todas las funciones que son utilizadas en los notebbooks descritos anteriormente. A excepción del algoritmo TREPAN que se encuentra implementado en [SurrogateTrees](/SurrogateTrees).

## Construido con

A continuación se describen las herramientas y librerías utilizadas para eñ desarrollo del proyecto:

* [Python](https://www.python.org/) - como lenguaje de programación utilizado para el desarrollo de código
* [Anaconda](https://www.anaconda.com/) - como gestor de variables de entorno y de todas las librerías necesarias para el proyecto.
* [Jupyter-Lab](https://www.anaconda.com/) - como entorno de desarrollo de código.
* [Pandas](https://pandas.pydata.org/) - librería que nos proporciona estructuras de datos fáciles de usar y con un alto rendimiento, permitiéndonos trabajar fácilmente con nuestros datos.
* [Numpy](https://numpy.org/) - librería que nos facilita las operaciones realizadas con vectores y matrices de cara a trabajar con Python.
* [TensorFlow](https://www.tensorflow.org/) - librería de aprendizaje automático para el desarrollo de modelos de Machine Learning.
* [Keras](https://keras.io/) - API para redes neuronales que se ejecuta sobre TensorFlow, abstrayendo muchos procesos a un alto nivel. 
* [Scikit-learn](https://scikit-learn.org/stable/) - librería utilizada para la minería y análisis de datos y el aprendizaje automático. 
* [Matplotlib](https://matplotlib.org/) - librería que permite generar gráficas y figuras para la visualización de datos y resultados. 
* [SciPy](https://planet.scipy.org/) - librería para generar resultados estadísticos a partir de los datos con los que trabajamos.
* [Impyute](https://impyute.readthedocs.io/en/master/) - librería con diversos algoritmos de imputación de datos faltantes desarrollada para Python.
* [Missingpy](https://pypi.org/project/missingpy/) - librería para el relleno de datos perdidos con Python, basada en scikit-learn para simplificar su funcionamiento.
* [Imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html) - librería que contiene diversos algoritmos de balanceo de clases dentro de un conjunto de datos para su correcto funcionamiento al aplicar técnicas de Machine Learning.
* [Shap](https://shap.readthedocs.io/en/latest/) - librería que permite explicar los resultados obtenidos por cualquier modelo de Machine Learning.
