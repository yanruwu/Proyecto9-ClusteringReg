# Proyecto 9: Clustering y Modelos de Regresión 🧩📈

*Descubriendo patrones y optimizando estrategias para maximizar el éxito empresarial*

## Descripción del Proyecto

En este proyecto, asumo el rol de científico de datos para una empresa de comercio global, con el objetivo de comprender a fondo su base de clientes, productos y operaciones. La empresa enfrenta el desafío de tomar decisiones estratégicas basadas en datos para maximizar beneficios y optimizar procesos. A través del análisis de un conjunto de datos que incluye información sobre ventas, envíos, costos y beneficios, este proyecto busca extraer insights significativos que orienten decisiones clave.

### **Objetivos Clave** 🎯
1. **Segmentación:**  
   Identificar patrones en los datos mediante técnicas de clustering para agrupar clientes y productos según características relevantes como comportamiento de compra o rentabilidad.
   
2. **Modelos de Predicción por Segmentos:**  
   Diseñar modelos de regresión específicos para cada grupo, analizando los factores más relevantes que influyen en las ventas y beneficios dentro de cada segmento.

3. **Acción Estratégica Basada en Insights:**  
   Usar los resultados del análisis para diseñar estrategias que:
   - Enfoquen recursos en los segmentos más rentables.
   - Mejoren el desempeño de los segmentos menos rentables mediante intervenciones específicas.

Este análisis permitirá a la empresa optimizar estrategias de marketing, ajustar políticas de precios y descuentos, y tomar decisiones basadas en datos para un impacto tangible.


## **Progreso del Proyecto** 🚀

### **Planteamiento de la estrategia y EDA** 🔍

Primero se realizó una exploración inicial de los datos, de tal forma que se pudiera entender la naturaleza de estos y la información disponible.

Los datos consistían en información de las compras de los clientes: información del pedido e información del cliente. Con esta información se tomaría una decisión de cómo agruparlos.

### **Preprocesado y Clustering** 🔧

Según la estrategia a tomar, se realizó un preprocesamiento en base a esta y se emplearon diferentes tipos de modelo de clusterización, como KMeans y DBSCAN, para agrupar nuestros datos en grupos con características diferenciadas.

### **Entrenamiento de Modelos de Regresión** 🤖

Por último se entrenaron diferentes modelos sobre cada uno de los datasets diferentes surgidos de la clusterización, lo cual nos permitió obtener insights valiosos para la toma de decisiones dentro de la empresa.


## **Estructura del Proyecto** 🗂️


## Estructura del Proyecto 🗂️
```
├── datos/                       # Archivos CSV y datos en crudo
│   ├── clean.pkl                # Datos limpios en formato pickle
│   ├── clusters.pkl             # Labels de los clusters generados
│   └── Global_Superstore.csv    # Datos originales del proyecto
├── img/                         # Imágenes
├── notebooks/                   # Notebooks Jupyter para EDA y desarrollo de modelos
│   ├── EDA.ipynb                # Análisis exploratorio de los datos
│   ├── Clustering/              # Carpeta con los procesos de clusterización
│   │   ├── 1.1-prep-cluster.ipynb
│   │   └── 1.2-prep-cluster.ipynb
│   ├── Regression/              # Carpeta con código de modelos de regresión
│   │   └── 2-reg.ipynb
├── src/                         # Código fuente principal para el preprocesamiento y modelado
│   ├── support_clustering.py    # Funciones de soporte para clustering
│   ├── support_eda.py           # Funciones de soporte para EDA
│   ├── support_prep.py          # Funciones para la preparación y limpieza de los datos
│   ├── support_reg.py           # Funciones de soporte para modelos de regresión
│   └── support_test_stats.py    # Funciones de soporte para tests estadísticos
├── environment.yml              # Archivo de configuración para gestionar dependencias del entorno
└── README.md                    # Documentación del proyecto
```


## Instalación y Requisitos ⚙️

Para configurar el entorno de desarrollo y asegurarse de que todas las dependencias necesarias estén instaladas, se deben seguir estos pasos:

### Requisitos

- Python 3.8 o superior 🐍
- [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (opcional, pero recomendado)

### Paquetes Necesarios

El proyecto utiliza los siguientes paquetes:

- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/): Para la manipulación y análisis de datos.
- [`matplotlib`](https://matplotlib.org/stable/users/index.html): Para la visualización de datos.
- [`seaborn`](https://seaborn.pydata.org/): Para visualización estadística de datos.
- [`category-encoders`](https://contrib.scikit-learn.org/category_encoders/): Para la codificación de variables categóricas.
- [`scikit-learn`](https://scikit-learn.org/stable/): Para la implementación de modelos de machine learning y herramientas de preprocesamiento.
- [`scipy`](https://scipy.org/): Para funciones avanzadas de estadística, álgebra lineal y optimización.
- [`shap`](https://shap.readthedocs.io/en/latest/): Para la interpretabilidad de modelos a través de valores SHAP.
- [`xgboost`](https://xgboost.readthedocs.io/en/stable/): Para la implementación del algoritmo de Gradient Boosting eficiente y optimizado.



## **Resultados** 📊

### Clustering

#### Iteración 1
Se aplicó un encoding por frecuencias a las variables categóricas y se ajustó un modelo **DBSCAN**. Los parámetros se configuraron basándonos en visualizaciones de las distancias a los k-ésimos vecinos. Este modelo produjo 3 clusters diferenciados principalmente por la región de las ventas, un resultado predecible debido al encoding utilizado, pero poco útil para el análisis empresarial.

#### Iteración 2
Se cambió a un **target encoding**, utilizando el *profit* como variable objetivo. Esto permitió diferenciar las categorías según el beneficio que aportaban a la empresa. A partir de este encoding, se evaluaron dos estrategias de clustering:

- **KMeans:** Usando un *ElbowVisualizer* y un *PCA*, se determinó que el número óptimo de clusters era 2. Estos clusters se diferenciaban principalmente por el *profit* de los registros, con influencia adicional de otras características correlacionadas.

- **DBSCAN:** Aplicando la misma estrategia de la iteración 1 pero con el nuevo encoding, el modelo produjo un solo cluster y múltiples registros catalogados como ruido, lo que no resultó útil para el análisis.

#### Resultado del Clustering
Optamos por los 2 clusters formados por **KMeans** en la Iteración 2. Este modelo ofrece una segmentación útil para identificar grupos de registros con diferentes niveles de rentabilidad. Estos clusters se diferencian por el profit, y observando las variables se aprecia una diferencia clave en los tipos de productos que se venden, los cuales tienen un impacto significativo en los beneficios de la empresa.

### Modelos de Regresión
El objetivo de nuestro análisis será comprobar qué factores influyen positiva o negativamente en los beneficios obtenidos en las ventas. Para ello nos enfocaremos en comrprobar los beneficios en cada uno de los clusters.

#### Preprocesamiento y elección de variables

- Se comprobaron las relaciones de las diferentes variables con el beneficio y se compararon los dos clusters con más detalle.

- Se empleó un **target encoding** sobre el profit, como anteriormente, además de un **onehot encoding** para aquellas variables categóricas que no presentaban diferencias significativas entre categorías internas.

- Se realizó una estandarización usando un **robust scaler**, debido a la naturaleza de los datos, los cuales no se asemejaban a distribuciones normales y presentaban outliers univariados.

- Se identificaron y eliminaron outliers con el método **Isolation Forest**, que permitió encontrar aquellos registros más aislados que influirían negativamente a la hora de entrenar nuestros modelos.

- Se realizaron dos iteraciones, donde en la primera mantuvimos todas las variables disponibles, y tras comprobar el bajo impacto de algunas de ellas sobre el modelo, se eliminaron para tener un modelo más simple pero igual de efectivo.

#### Entrenamiento de Modelos
Para ambos clusters se entrenaron 3 modelos distintos: **Random Forest**, **Gradient Boosting** y **XGBoost**. 
- Para el primer cluster los mejores resultados se obtuvieron usando Random Forest:

|       | R2       | MAE        | RMSE       |
|-------|----------|------------|------------|
| Train | 0.822562 | 23.820745  | 48.297832  |
| Test  | 0.780632 | 24.712413  | 53.136923  |


- Para el segundo cluster los mejores resultados se obtuvieron usando Gradient Boosting:

|       | R2       | MAE        | RMSE       |
|-------|----------|------------|------------|
| Train | 0.725162 | 157.392946 | 219.213188 |
| Test  | 0.612236 | 187.286322 | 334.575950 |

Más detalles en el notebook asociado.

## **Conclusiones** 💡

### **Cluster 0 (Bajos Beneficios)**
- **Descuentos:** Principal factor negativo en los beneficios, por lo que deberían reducirse o eliminarse.
- **Ventas:** Ventas altas impactan negativamente debido a los bajos márgenes de productos como material de oficina.
- **Costos de Envío:** Altos costos logísticos también afectan a los beneficios, indicando necesidad de optimización.

**Estrategia:** Reducir descuentos, optimizar logística y evaluar productos de bajo margen.


### **Cluster 1 (Altos Beneficios)**
- **Ventas:** A mayores ventas, mayor beneficio, gracias a los buenos márgenes de productos tecnológicos y muebles.
- **Costos de Envío:** Impactan menos en los beneficios, indicando buena optimización logística.
- **Descuentos:** Tienen un impacto negativo moderado, pero pueden incentivar ventas sin afectar significativamente los márgenes.

**Estrategia:** Maximizar ventas con descuentos moderados y mantener la eficiencia logística.

### **Preguntas Clave y Respuestas**

#### **1. ¿Cómo podemos agrupar a los clientes o productos de manera significativa?**
Se realizaron agrupaciones (clusters) basadas en variables como rentabilidad (`profit`), ventas (`sales`), costos de envío y descuentos. Esto permitió identificar:
- **Cluster 0:** Productos de bajo margen y clientes menos rentables.
- **Cluster 1:** Productos y clientes con alta rentabilidad, principalmente en tecnología y muebles.


#### **2. ¿Qué factores son más relevantes para predecir el beneficio o las ventas dentro de cada grupo?**
- **Cluster 0:** 
  - Los descuentos y los costos de envío son los principales factores negativos para los beneficios.
  - Las ventas altas tienden a generar márgenes negativos debido a productos de bajo margen, como material de oficina.
- **Cluster 1:** 
  - Las ventas son el principal impulsor del beneficio, mientras que los descuentos tienen un impacto moderado.
  - Los costos de envío afectan menos, gracias a una logística más eficiente.


#### **3. ¿Cómo podemos utilizar estos insights para tomar decisiones estratégicas?**
- **Cluster 0:** Reducir descuentos y optimizar la logística para mejorar márgenes. Considerar ajustes de precios o descontinuar productos poco rentables.
- **Cluster 1:** Enfocar esfuerzos en maximizar ventas con descuentos estratégicos y mantener la optimización logística.

