# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 05:57:25 2023
Desarrollo para evaluacion de impacto con metodos de machine learning
@author: JULIAN FLOREZ
"""

import pandas as pd
import os


# Ruta de la carpeta donde se encuentran los archivos
ruta_carpeta = "E:\\ESAP\\Estudio_Impacto"

# Función para leer un archivo de Excel y devolver un dataframe
def leer_excel(nombre_archivo):
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo + ".xlsx")
    try:
        # Especificar tipos de datos solo para ciertas columnas
        tipos_columnas = {'Código Departamento': str, 'Código Entidad': str}
        return pd.read_excel(ruta_completa, dtype=tipos_columnas)
    except Exception as e:
        print(f"No se pudo leer el archivo '{nombre_archivo}': {e}")
        return None

# Crear dataframes individual
df_Tabla_Maestra = leer_excel("DataEval_2_nombre_largo")


# Lista de años que queremos filtrar
anios_deseados = [2016, 2017, 2018, 2019, 2020, 2021]

# Filtrar el DataFrame para obtener solo las filas con los años deseados
Tabla_Maestra = df_Tabla_Maestra[df_Tabla_Maestra['Año'].isin(anios_deseados)]

# Este método toma automáticamente el promedio de los valores de las filas superior e inferior.
Tabla_Maestra['Movilizacion_Recursos'] = Tabla_Maestra['Movilizacion_Recursos'].interpolate(method='linear')

# Imprimir los primeros 6 registros de 'Movilizacion_Recursos'
print(Tabla_Maestra['Movilizacion_Recursos'].head(6))

valores_a_eliminar = [88001, 91263, 91405, 91407, 91430, 91460, 91530, 91536, 91669, 91798, 94663, 94883, 94884, 94885, 94886, 94887, 94888, 97511, 97777, 97889]
Tabla_Maestra = Tabla_Maestra[~Tabla_Maestra['DEPMUN'].isin(valores_a_eliminar)]

# Imprimir los primeros 6 registros de 'Movilizacion_Recursos'
print(Tabla_Maestra['Valor_Agreg_PerCapiNac'].head(6))

# Imprimir los primeros 6 registros de 'Movilizacion_Recursos'
print(Tabla_Maestra['PctgasTotalInversion'].head(6))

#Se infieren los datos para el año 2021 a partir de media movil
def impute_nan_with_rolling_mean(row, window=5, min_periods=1):
    if pd.isna(row['Valor_Agregado_PerCapita']):
        # Calculamos la media móvil utilizando los datos hasta el punto del NaN
        rolling_mean = Tabla_Maestra['Valor_Agregado_PerCapita'].rolling(window=window, min_periods=min_periods).mean().shift(2)
        return rolling_mean[row.name]  # Usamos row.name para obtener el índice correcto de la serie
    else:
        return row['Valor_Agregado_PerCapita']

# Aplicar la función a cada fila
Tabla_Maestra['Valor_Agregado_PerCapita'] = Tabla_Maestra.apply(impute_nan_with_rolling_mean, axis=1)


#Se infieren los datos para el año 2021 a partir de media movil
def impute_nan_with_rolling_mean(row, window=5, min_periods=1):
    if pd.isna(row['Valor_Agreg_PerCapiNac']):
        # Calculamos la media móvil utilizando los datos hasta el punto del NaN
        rolling_mean = Tabla_Maestra['Valor_Agreg_PerCapiNac'].rolling(window=window, min_periods=min_periods).mean().shift(2)
        return rolling_mean[row.name]  # Usamos row.name para obtener el índice correcto de la serie
    else:
        return row['Valor_Agreg_PerCapiNac']

# Aplicar la función a cada fila
Tabla_Maestra['Valor_Agreg_PerCapiNac'] = Tabla_Maestra.apply(impute_nan_with_rolling_mean, axis=1)

#Se infieren los datos para el año 2021 a partir de media movil
def impute_nan_with_rolling_mean(row, window=5, min_periods=1):
    if pd.isna(row['PctIngCorrFuncionamiento']):
        # Calculamos la media móvil utilizando los datos hasta el punto del NaN
        rolling_mean = Tabla_Maestra['PctIngCorrFuncionamiento'].rolling(window=window, min_periods=min_periods).mean().shift(2)
        return rolling_mean[row.name]  # Usamos row.name para obtener el índice correcto de la serie
    else:
        return row['PctIngCorrFuncionamiento']

# Aplicar la función a cada fila
Tabla_Maestra['PctIngCorrFuncionamiento'] = Tabla_Maestra.apply(impute_nan_with_rolling_mean, axis=1)

#Se infieren los datos para el año 2021 a partir de media movil
def impute_nan_with_rolling_mean(row, window=5, min_periods=1):
    if pd.isna(row['PctIngCorrRecursosPropios']):
        # Calculamos la media móvil utilizando los datos hasta el punto del NaN
        rolling_mean = Tabla_Maestra['PctIngCorrRecursosPropios'].rolling(window=window, min_periods=min_periods).mean().shift(2)
        return rolling_mean[row.name]  # Usamos row.name para obtener el índice correcto de la serie
    else:
        return row['PctIngCorrRecursosPropios']

# Aplicar la función a cada fila
Tabla_Maestra['PctIngCorrRecursosPropios'] = Tabla_Maestra.apply(impute_nan_with_rolling_mean, axis=1)

#Se infieren los datos para el año 2021 a partir de media movil
def impute_nan_with_rolling_mean(row, window=5, min_periods=1):
    if pd.isna(row['PctIngresoTransferencias']):
        # Calculamos la media móvil utilizando los datos hasta el punto del NaN
        rolling_mean = Tabla_Maestra['PctIngresoTransferencias'].rolling(window=window, min_periods=min_periods).mean().shift(2)
        return rolling_mean[row.name]  # Usamos row.name para obtener el índice correcto de la serie
    else:
        return row['PctIngresoTransferencias']

# Aplicar la función a cada fila
Tabla_Maestra['PctIngresoTransferencias'] = Tabla_Maestra.apply(impute_nan_with_rolling_mean, axis=1)

#Se infieren los datos para el año 2021 a partir de media movil
def impute_nan_with_rolling_mean(row, window=5, min_periods=1):
    if pd.isna(row['PctgasTotalInversion']):
        # Calculamos la media móvil utilizando los datos hasta el punto del NaN
        rolling_mean = Tabla_Maestra['PctgasTotalInversion'].rolling(window=window, min_periods=min_periods).mean().shift(2)
        return rolling_mean[row.name]  # Usamos row.name para obtener el índice correcto de la serie
    else:
        return row['PctgasTotalInversion']

# Aplicar la función a cada fila
Tabla_Maestra['PctgasTotalInversion'] = Tabla_Maestra.apply(impute_nan_with_rolling_mean, axis=1)

# Eliminar columnas con datos vacíos (NaN)
df_filtrada_sin_vacios = Tabla_Maestra.dropna(axis=1)

Tabla_maestra = df_filtrada_sin_vacios

# Eliminar columnas cuyos nombres contienen '_IND'
Tabla_maestra = Tabla_maestra.loc[:, ~Tabla_maestra.columns.str.contains('_IND')]


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Separar la variable dependiente (este paso ya lo hiciste previamente)
X = Tabla_maestra.drop(columns=['Movilizacion_Recursos', 'DEPMUN', 'VA', 'VAPC'])

# Normalizar las variables independientes
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=4)
componentes_principales = pca.fit_transform(X_normalizado)

# Crear un DataFrame con los componentes principales
df_pca = pd.DataFrame(data=componentes_principales, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Varianza explicada por cada componente
varianza_explicada = pca.explained_variance_ratio_
print("Varianza explicada por cada componente:", varianza_explicada)

# Graficar el primer componente contra el segundo
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Componentes 1 y 2')
plt.grid(True)
plt.show()

# Sumar acumulativamente la varianza explicada
varianza_acumulada = np.cumsum(varianza_explicada)

# Graficar la varianza explicada y la varianza acumulada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.5, align='center', label='Varianza individual explicada')
plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid', label='Varianza acumulada explicada')
plt.ylabel('Proporción de Varianza Explicada')
plt.xlabel('Componentes Principales')
plt.legend(loc='best')
plt.title('Varianza Explicada por Diferentes Componentes Principales')
plt.show()

# Obtener las cargas (loadings) de los componentes principales
loadings = pca.components_

# Convertir las cargas a un DataFrame para una mejor visualización
df_loadings = pd.DataFrame(loadings.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.columns)


# Mostrar las cargas de los componentes
print(df_loadings)

# Visualizar las contribuciones de las variables a los componentes principales
plt.figure(figsize=(12, 8))
sns.heatmap(df_loadings, annot=True, cmap='coolwarm')
plt.title('Cargas de las Variables en los Componentes Principales')
plt.show()

# Filtrar los valores en la columna PC1 que son mayores o iguales a 0.1
variables_independientes = df_loadings[df_loadings['PC1'] >= 0.1].index.tolist()

X1 = Tabla_maestra[variables_independientes]
Y1 = Tabla_maestra['Movilizacion_Recursos']



##############Clustering###################
os.environ['OMP_NUM_THREADS'] = '5'

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
import seaborn as sns

#Tomar la Tabla maestra para continuar
df = Tabla_maestra[['Movilizacion_Recursos', 'Ingre_Totales']]


# Normalizar los datos
df_norm = scale(df)

# Estimar el número óptimo de clústeres usando el método del codo
range_n_clusters = list(range(2, 11))
distortions = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df_norm)
    distortions.append(sum(np.min(cdist(df_norm, kmeans.cluster_centers_, 'euclidean'), axis=1)) / df_norm.shape[0])

# Graficar el método del codo
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Estimar el número óptimo de clústeres usando el método de la silueta
silhouette_avg_metrics = []

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(df_norm)
    silhouette_avg = silhouette_score(df_norm, cluster_labels)
    silhouette_avg_metrics.append(silhouette_avg)

# Graficar el método de la silueta
colors = plt.cm.rainbow(np.linspace(0, 1, len(range_n_clusters)))
plt.figure(figsize=(10, 5))
bars = plt.bar(range_n_clusters, silhouette_avg_metrics, color=colors)

# Mostrar los valores en las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Method for Optimal Number of Clusters')
plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"))
plt.show()

# Clustering con KMeans
kmeans_model = KMeans(n_clusters=5)  # Cambiar según el resultado del método del codo o de la silueta
kmeans_model.fit(df_norm)
df['Cluster'] = kmeans_model.labels_

# Resumen de clústeres
summary_clusters = df.groupby('Cluster').mean()

# Preparar datos para el gráfico final
data_long = pd.melt(df, id_vars=['Cluster'], var_name='Feature', value_name='Value')

# Crear el gráfico final
custom_palette = sns.color_palette("husl", n_colors=data_long['Cluster'].nunique())
sns.lineplot(data=data_long, x='Feature', y='Value', hue='Cluster', estimator=np.mean, err_style='bars', palette=custom_palette)
plt.show()


############SOM

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# 1. Leer los datos
productores = X1
productores01 = productores.iloc[:, :37]  # Seleccionar las primeras 14 columnas

# 2. Escalar los datos
scaler = StandardScaler()
productores_sc = scaler.fit_transform(productores01)

# 3. Crear y entrenar el SOM
som = MiniSom(4, 4, productores_sc.shape[1], sigma=1.0, learning_rate=0.5, topology='hexagonal', random_seed=1000)
som.train_random(productores_sc, 200)

# 4. Visualizar los resultados del SOM
# (La visualización dependerá de qué tipo de visualización necesitas; aquí hay un ejemplo simple)
plt.figure(figsize=(7, 7))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # MiniSom 2.2.3 or higher required
plt.colorbar()
plt.show()

# Opcional: 5. Realizar clustering jerárquico sobre los códigos SOM
clusterer = AgglomerativeClustering(n_clusters=5)
clusters = clusterer.fit_predict(som.get_weights().reshape(-1, productores_sc.shape[1]))


# Calculamos las neuronas ganadoras para cada muestra
winners = np.array([som.winner(x) for x in productores_sc])

# Convertimos los índices 2D (x, y) de los ganadores a un índice 1D
winner_indices = winners[:, 0] * som._weights.shape[1] + winners[:, 1]

# Asignamos a cada muestra su etiqueta de cluster correspondiente
productores['cluster'] = clusters[winner_indices]

# Calcular la suma total de cada variable en toda la tabla
suma_total = productores01.sum()

# Suma de cada variable dentro de cada cluster
suma_por_cluster = productores01.groupby(productores['cluster']).sum()

# Calcular la proporción de cada variable en cada cluster respecto a la suma total
proporcion_por_cluster = suma_por_cluster.divide(suma_total, axis=1)

# Ahora `proporcion_por_cluster` contiene la proporción de cada variable para cada cluster
print(proporcion_por_cluster)
# Guardar los resultados
#productores.to_csv('Productores_Class.csv', index=False)

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
# Número de variables
categories = list(proporcion_por_cluster.columns)
N = len(categories)

# Ángulos para cada eje en el gráfico de radar
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # completar el círculo

# Función para crear un gráfico de radar para cada cluster
def make_spider(row, title, color):
    values = proporcion_por_cluster.iloc[row].values.flatten().tolist()
    values += values[:1]
    ax = plt.subplot(3, 2, row+1, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
    plt.title(title, size=11, color=color, y=1.1)

# Tamaño de la figura
plt.figure(figsize=(10, 15))

# Crear un gráfico de radar para cada cluster
for i in range(proporcion_por_cluster.shape[0]):
    make_spider(i, f'Cluster {i+1}', 'b')

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.show()




####Modelado Clasificacion de los 5 clases de Kmeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df_filtrada_sin_vacios = Tabla_maestra

# Seleccionar la variable 'movilizacion_de_recursos' para aplicar KMeans
data_clustering = df_filtrada_sin_vacios[['Movilizacion_Recursos', 'Ingre_Totales']]

# Estandarizar los datos
scaler = StandardScaler()
data_clustering_scaled = scaler.fit_transform(data_clustering)

# Aplicar KMeans con 5 centroides
kmeans = KMeans(n_clusters=5, random_state=42)
df_filtrada_sin_vacios['Clusterin_Kmeans'] = kmeans.fit_predict(data_clustering_scaled)

# Mostrar los primeros registros
print(df_filtrada_sin_vacios.head())

df_filtrada_sin_vacios.to_excel('20231211Clusteres.xlsx', index=False)


# Eliminar las columnas especificadas
df_cleaned = df_filtrada_sin_vacios.drop(columns=["DEPMUN", "VA", "VAPC", "Movilizacion_Recursos"])


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Preparación de los datos
X = df_cleaned.drop(columns=["Clusterin_Kmeans"])
y = df_cleaned["Clusterin_Kmeans"]


# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de Árbol de Decisiones
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Modelo de Bosque Aleatorio
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Métricas para el Árbol de Decisiones
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
feature_importance_dt = dt_model.feature_importances_

# Métricas para el Bosque Aleatorio
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
feature_importance_rf = rf_model.feature_importances_

# Imprimir resultados
print("Árbol de Decisiones - Confusión Matrix:")
print(conf_matrix_dt)
print("Árbol de Decisiones - Precisión:", accuracy_dt)
print("Árbol de Decisiones - Importancia de las Variables:")
for feature, importance in zip(X.columns, feature_importance_dt):
    print(f"{feature}: {importance:.2%}")

print("\nBosque Aleatorio - Confusión Matrix:")
print(conf_matrix_rf)
print("Bosque Aleatorio - Precisión:", accuracy_rf)
print("Bosque Aleatorio - Importancia de las Variables:")
for feature, importance in zip(X.columns, feature_importance_rf):
    print(f"{feature}: {importance:.2%}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



# Balanceando los datos
df_balanced = pd.concat([resample(df_cleaned [df_cleaned ["Clusterin_Kmeans"] == cluster],
                                  replace=True, 
                                  n_samples=df_cleaned ["Clusterin_Kmeans"].value_counts().max(),
                                  random_state=42) 
                         for cluster in df_cleaned ["Clusterin_Kmeans"].unique()])

X_balanced = df_balanced.drop(columns=["Clusterin_Kmeans"])
y_balanced = df_balanced["Clusterin_Kmeans"]

# Seleccionar las variables independientes
variables_independientes = [
    'Ingre_Tribut_Per_Capita',
    'Ingre_Per_Capi_Imp_Predial',
    'Ingre_No_Tribut_Per_Capi',
    'PctIngresoTransferencias',
    'Año',
    'PctgasTotalInversion',
    'Gasto_Corrientes',
    'Ingre_Corrientes_Per_Capi',
    'Ingre_No_Tribut',
    'AfiliadoRegimenContributivo',
    'Proposito_General',
    'Ingre_Totales_Per_Capita',
    'Funcionamiento',
    'Trans_Ingresos_Corrientes',
    'DeficAhorro_corriente',
    'Salud',
    'Ingre_Tributarios',
    'CrediInterno_Externo',
    'Penet_Banda_Ancha',
    'PIFAT'
]

# Separar la variable dependiente
X_balanced1 = X_balanced[variables_independientes]

X_balanced = X_balanced1


# Dividiendo los datos balanceados en conjuntos de entrenamiento y prueba
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# Definir y entrenar AdaBoost
adaboost_model_bal = AdaBoostClassifier(random_state=42)
adaboost_model_bal.fit(X_train_bal, y_train_bal)

# Definir y entrenar GBM
gbm_model_bal = GradientBoostingClassifier(random_state=42)
gbm_model_bal.fit(X_train_bal, y_train_bal)

# Definir y entrenar el modelo de Stacking
base_models = [
    ('gbm', GradientBoostingClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('rf', RandomForestClassifier(random_state=42))
]
stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=42))
stacking_model.fit(X_train_bal, y_train_bal)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle

def evaluate_model_multiclass(model, X_test, y_test, model_name, classes):
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Binarizar el y_test para multiclase
    y_test_binarized = label_binarize(y_test, classes=classes)

    # Curva ROC y AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular la micro-media de la curva ROC y el área bajo la curva
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Graficar la curva ROC para cada clase y la micro-media
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Matriz de confusión y exactitud
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Confusion Matrix:\n{cm}")
    print(f"{model_name} Accuracy: {accuracy}\n")


# Lista de clases únicas en tu conjunto de datos
classes = np.unique(y_test_bal)

# Evaluar modelos para clasificación multiclase
evaluate_model_multiclass(adaboost_model_bal, X_test_bal, y_test_bal, 'AdaBoost', classes)
evaluate_model_multiclass(gbm_model_bal, X_test_bal, y_test_bal, 'Gradient Boosting Machine', classes)
evaluate_model_multiclass(stacking_model, X_test_bal, y_test_bal, 'Stacking Model', classes)

import joblib

# Guardar el modelo entrenado en un archivo
joblib.dump(gbm_model_bal, 'gbm_model_bal.pkl')

# Mensaje de confirmación
print("Modelo guardado como 'gbm_model_bal.pkl'")

# Cargar el modelo
gbm_model_loaded = joblib.load('gbm_model_bal.pkl')

################################
####Modelo cuantitativo de crecimiento


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Cargar y preparar los datos
# Seleccionar las variables independientes para evaluacion de impacto
variables_independientes = [
    'Ingre_Tribut_Per_Capita',
    'Ingre_Per_Capi_Imp_Predial',
    'Ingre_No_Tribut_Per_Capi',
    'PctIngresoTransferencias',
    'Año',
    'PctgasTotalInversion',
    'Gasto_Corrientes',
    'Ingre_Corrientes_Per_Capi',
    'Ingre_No_Tribut',
    'AfiliadoRegimenContributivo',
    'Proposito_General',
    'Ingre_Totales_Per_Capita',
    'Funcionamiento',
    'Trans_Ingresos_Corrientes',
    'DeficAhorro_corriente',
    'Salud',
    'Ingre_Tributarios',
    'CrediInterno_Externo',
    'Penet_Banda_Ancha',
    'PIFAT',
    'Movilizacion_Recursos'
]

# Separar la variable dependiente
cuantitativo = Tabla_maestra[variables_independientes]
cuantitativo.to_excel('cuantitativo.xlsx')

X = cuantitativo.drop(columns=["Movilizacion_Recursos"])
y = cuantitativo["Movilizacion_Recursos"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Construir el modelo
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10)

# Evaluar el modelo
loss = model.evaluate(X_test_scaled, y_test)
print(loss)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Realizar predicciones con el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir las métricas
print("Error Cuadrático Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)
print("Coeficiente de Determinación (R^2):", r2)

X1.to_excel('Tabla_Variables.xlsx')
Y1.to_excel('Tabla_Dependiente.xlsx')
Tabla_maestra.to_excel('Tabla_Kmeans.xlsx')
df.to_excel('Tabla_SOM.xlsx')
Tabla_maestra.to_excel('Tabla_maestra.xlsx')

#Evaluacion del impacto
XNPIFAT = cuantitativo.drop(columns=["Movilizacion_Recursos"])

# Configurar todos los valores en la columna 'PIFAT' a cero
XNPIFAT['PIFAT'] = 0

# Escalar X con el scaler ya entrenado
X_scaled = scaler.transform(XNPIFAT)

# Realizar predicciones con el modelo
y_pred = model.predict(X_scaled)

# Crear un nuevo DataFrame con las predicciones agregadas
X_with_predictions = XNPIFAT.copy()
X_with_predictions['Prediction'] = y_pred.flatten()

X_with_predictions.to_excel('SinPIFAT.xlsx')
##########################################
#Aplicando el proyecto desde el año 2019

XPIFAT2019= cuantitativo.drop(columns=["Movilizacion_Recursos"])
# Configurar todos los valores en la columna 'PIFAT' igual a 1 en el 2019
XPIFAT2019.loc[XPIFAT2019['Año'] == 2019, 'PIFAT'] = 1


# Escalar X con el scaler ya entrenado
X_scaled = scaler.transform(XPIFAT2019)

# Realizar predicciones con el modelo
y_pred = model.predict(X_scaled)

# Crear un nuevo DataFrame con las predicciones agregadas
X_with_predictions2019 = XPIFAT2019.copy()
X_with_predictions2019['Prediction'] = y_pred.flatten()

X_with_predictions2019.to_excel('PIFAT2019.xlsx')
##########################################
#Aplicando el proyecto desde el año 2018

XPIFAT2018 = cuantitativo.drop(columns=["Movilizacion_Recursos"])
# Configurar todos los valores en la columna 'PIFAT' igual a 1 en el 2019
XPIFAT2018.loc[XPIFAT2018['Año'] == 2019, 'PIFAT'] = 1
XPIFAT2018.loc[XPIFAT2018['Año'] == 2018, 'PIFAT'] = 1

# Escalar X con el scaler ya entrenado
X_scaled = scaler.transform(XPIFAT2018)

# Realizar predicciones con el modelo
y_pred = model.predict(X_scaled)

# Crear un nuevo DataFrame con las predicciones agregadas
X_with_predictions2018 = XPIFAT2018.copy()
X_with_predictions2018['Prediction'] = y_pred.flatten()

X_with_predictions2018.to_excel('PIFAT2018.xlsx')

##########################################
#Aplicando el proyecto desde el año 2017

XPIFAT2017 = cuantitativo.drop(columns=["Movilizacion_Recursos"])
# Configurar todos los valores en la columna 'PIFAT' igual a 1 en el 2019
XPIFAT2017.loc[XPIFAT2017['Año'] == 2019, 'PIFAT'] = 1
XPIFAT2017.loc[XPIFAT2017['Año'] == 2018, 'PIFAT'] = 1
XPIFAT2017.loc[XPIFAT2017['Año'] == 2017, 'PIFAT'] = 1

# Escalar X con el scaler ya entrenado
X_scaled = scaler.transform(XPIFAT2017)

# Realizar predicciones con el modelo
y_pred = model.predict(X_scaled)

# Crear un nuevo DataFrame con las predicciones agregadas
X_with_predictions2017 = XPIFAT2017.copy()
X_with_predictions2017['Prediction'] = y_pred.flatten()

X_with_predictions2017.to_excel('PIFAT2017.xlsx')

#Evaluacion del impacto

# Escalar X con el scaler ya entrenado
X_scaled = scaler.transform(X)

# Realizar predicciones con el modelo
y_pred = model.predict(X_scaled)

# Crear un nuevo DataFrame con las predicciones agregadas
X_with_predictions= X.copy()
X_with_predictions['Prediction'] = y_pred.flatten()

X_with_predictions.to_excel('PIFAT021.xlsx')
