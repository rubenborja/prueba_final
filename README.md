# PRUEBA FINAL

Para el desarrollo de este proyecto de un clasificador de tipos de carnes y el entrenamiento se detalla a continuación los pasos:

Se utilizo jupyter notebook (Python).

Se empieza por llamar a las librerías necesarias (tensorflow, numpy, os, matplotlib.pyplot, pathlib, sklearn, seaborn, scikit-image).

Obtener la ruta del dataset.
 
Dentro del proyecto se obtiene, clasifica y ordena a las imágenes a clase que pertenece y también se redimensiona las imágenes.

Preprocesar los datos. Esto implica normalizar los valores de píxel y remodelar los datos.

Se genera los datos de train y test. datos de la carpeta test no se utilizaron para el proceso de entrenamiento, únicamente para evaluar el modelo final.

Evaluar y predecir el modelo en el conjunto de datos.

A continuación, se describe el código, cabe indicar que tiene comentarios en las líneas de código según se fue utilizand.

Clasificación usando el model SVM

# Definir los parámetros para la búsqueda del grid
param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}

# Usar un subconjunto de datos más pequeño
X_train_subset = X_train[:1000]
y_train_subset = y_train[:1000]

# Dafinir el modelo SVM
svc = svm.SVC()

pred = clf.predict(X_test)

# Imprimir informe de clasificación
print("Informe de clasificación para - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, pred)))

Matriz de confusión del SVM

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz de confusión para el modelo SVM
cnf_matrix = confusion_matrix(y_test, pred)
print(cnf_matrix)

Clasificación usando el modelo KNN
# Definir el modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=3)

# Entrenar datos con el modelo KNN
knn_model.fit(X_train, y_train)

# Obtener predicciones para el modelo
predict_knn = best_model.predict(X_test)
print(predict_knn)

# Calcular matriz de confusión
knn_cnf_matrix = confusion_matrix(y_test, predict_knn)
print(knn_cnf_matrix)

# Matriz de confusión con Seaborn
class_names = np.unique(y_test)
sns.set(font_scale=1.2)  # tamaño de fuente 
plt.figure(figsize=(8, 6))  # Tamaño de gráfico

ax = sns.heatmap(knn_cnf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                 annot_kws={"fontsize": 14}, linewidths=0.5, linecolor="black")

ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# Etiqueta y titulo
ax.set_title("Matriz de confusión (KNN)", fontsize=16)
ax.set_xlabel("Predición etiqueta", fontsize=14)
ax.set_ylabel("Eqtiqueta", fontsize=14)

ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)

cbar = ax.figure.colorbar(ax.collections[0])
cbar.ax.tick_params(labelsize=12)

# Mostrar gráfico de matriz
plt.tight_layout()
plt.show()

la variable class_names contiene las ocho clases, es decir, CLASS_1 hasta CLASS_8.

MODELOS DE APRENDIZAJE DE CNN, KNN Y SVM UTILIZADOS PARA ESTA TAREA:

La red neuronal convolucional (CNN).- es un tipo de algoritmo de aprendizaje profundo que se usa comúnmente para tareas de clasificación de imágenes. Las CNN pueden aprender características de las imágenes mediante el uso de una serie de capas convolucionales y capas de agrupación.

K-Nearest Neighbors (KNN).- es un algoritmo de aprendizaje automático no paramétrico que se usa comúnmente para tareas de clasificación y regresión. KNN funciona encontrando los k ejemplos de entrenamiento más similares a un nuevo punto de datos y luego usa las etiquetas de esos ejemplos de entrenamiento para clasificar el nuevo punto de datos.

Support Vector Machine (SVM).- es un algoritmo de aprendizaje automático supervisado que se usa comúnmente para tareas de clasificación y regresión. SVM funciona al encontrar un hiperplano que separa las dos clases de datos de una manera que maximiza el margen entre las dos clases.

MATRIZ DE CONFUCIÓN
Una matriz de confusión es una tabla que se utiliza para evaluar el rendimiento de un modelo de clasificación. Muestra la cantidad de veces que el modelo clasificó correctamente cada clase, así como la cantidad de veces que clasificó incorrectamente cada clase.

CONCLUSIONES.

En este proyecto de clasificación de imágenes de carne, se usó matrices de confusión para ver qué tan bien el modelo puede distinguir entre diferentes tipos de carne. Por ejemplo, si el modelo intenta clasificar imágenes de carne, la matriz de confusión mostraría cuántas imágenes de cada tipo de carne clasificó correctamente el modelo y cuántas clasificó erróneamente como otro tipo de carne.

La precisión general del modelo se puede calcular dividiendo el número de predicciones correctas por el número total de predicciones.

La precisión del modelo para cada clase de datos se puede calcular dividiendo el número de predicciones correctas para esa clase por el número total de predicciones para esa clase.

Al analizar los resultados de una matriz de confusión, es posible identificar áreas en las que el modelo funciona correctamente y áreas en las que podría mejorarse. Esta información se puede utilizar para ajustar el modelo y mejorar su rendimiento general.
