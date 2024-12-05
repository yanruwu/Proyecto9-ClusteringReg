import seaborn as sns
import matplotlib.pyplot as plt
import math

def plot_col(data, x, y=None, dateplot="month"):
    """
    Genera gráficos para explorar visualmente las columnas de un DataFrame.

    Esta función permite graficar automáticamente columnas de un DataFrame dependiendo de su tipo de dato
    (numérico, categórico, fecha). Si se especifica `y`, se generarán gráficos que muestran la relación entre `x` y `y`.
    
    Parámetros:
    -----------
    data : pandas.DataFrame
        El DataFrame que contiene los datos a graficar.
        
    x : str o list
        Una columna o lista de columnas del DataFrame que se desean graficar. Si `x` es una cadena, se convierte automáticamente en una lista.
        
    y : str, opcional
        Nombre de la columna que se utilizará como la variable dependiente (`y`). Si se proporciona, los gráficos se crearán para mostrar `x` frente a `y`. 
        Si no se proporciona, se generarán gráficos individuales para cada columna en `x`.
    
    dateplot : str, opcional, por defecto "month"
        Determina cómo se deben graficar las columnas de tipo fecha. 
        Las opciones válidas son:
            - "month": Grafica los datos por mes.
            - "year": Grafica los datos por año.
        
    Retorno:
    --------
    None
        La función no devuelve ningún valor. En su lugar, muestra los gráficos generados.
        
    Ejemplos de Uso:
    ----------------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
            'edad': [25, 32, 40, 29],
            'salario': [50000, 60000, 70000, 55000],
            'departamento': ['HR', 'IT', 'IT', 'HR'],
            'fecha_ingreso': pd.to_datetime(['2020-01-15', '2019-03-20', '2018-07-12', '2021-05-10'])
        })
    >>> plot_col(df, x=['edad', 'salario', 'departamento', 'fecha_ingreso'], y='salario', dateplot='year')

    Descripción de la Funcionalidad:
    --------------------------------
    - La función primero verifica si `x` es un string y lo convierte en una lista si es necesario.
    - Luego, clasifica las columnas en tres categorías: numéricas, categóricas, y de tipo fecha.
    - Si `y` no está especificada:
        - Crea gráficos para cada columna en `x`:
            - Gráficos de histogramas para columnas numéricas.
            - Gráficos de conteo para columnas categóricas.
            - Gráficos de conteo por mes o año para columnas de tipo fecha.
    - Si `y` está especificada:
        - Grafica `x` frente a `y` dependiendo del tipo de las variables:
            - Diagrama de dispersión (`scatterplot`) si ambas son numéricas.
            - Diagrama de caja (`boxplot`) si `x` es categórico y `y` es numérico.
            - Gráfico de líneas (`lineplot`) si `x` es una fecha y `y` es numérico.
    - La función organiza automáticamente los subplots y ajusta la rotación de los ejes `x` cuando es necesario para mejorar la legibilidad.

    Advertencias:
    -------------
    - Si una columna no pertenece a un tipo soportado, se mostrará un mensaje de advertencia y no se graficará.
    - Si se crean más subplots de los necesarios, se eliminan los ejes sobrantes para evitar gráficos vacíos.

    """
    
    if type(x) == str:
        print("No es una lista. Convirtiendo en lista...")
        x = [x]
    print("Separando tipos de datos")
    num_cols = data[x].select_dtypes("number")
    cat_cols = data[x].select_dtypes("O")
    date_cols = data[x].select_dtypes("datetime")

    if not y:
        nrows = 2
        ncols = math.ceil(len(x)/2)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = (25,15), dpi =130)
        axes = axes.flat
        for i, col in enumerate(x):
            if col in num_cols:
                sns.histplot(data = data, x = col, ax = axes[i], bins = 20)
                axes[i].set_title(col)
            elif col in cat_cols:
                sns.countplot(data = data, x = col, ax = axes[i])
                axes[i].tick_params(axis='x', rotation=90)
                axes[i].set_title(col)
            elif col in date_cols:
                sns.countplot(data = data,
                               x = data[col].apply(lambda x: x.year) if dateplot == "year" else data[col].apply(lambda x: x.month),
                               ax = axes[i])
                axes[i].tick_params(axis='x', rotation=90)
                axes[i].set_title(f"{col}_{dateplot}")
            else:
                print(f"Advertencia: No se pudo graficar '{col}' frente a '{y}' ya que no es de un tipo soportado o la combinación no es válida.")
            
            axes[i].set_xlabel("")
            
        for j in range(len(x), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    else:
        nrows = math.ceil(len(x) / 2)
        ncols = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 25), dpi=130)
        axes = axes.flat

        for i, col in enumerate(x):
            # Caso 1: Numérico vs Numérico -> Scatterplot
            if col in num_cols and y in data.select_dtypes("number"):
                sns.scatterplot(data=data, x=col, y=y, ax=axes[i])
                axes[i].set_title(f'{col} vs {y}')
            # Caso 2: Categórico vs Numérico -> Boxplot o Violinplot
            elif col in cat_cols and y in data.select_dtypes("number"):
                sns.boxplot(data=data, x=col, y=y, ax=axes[i])
                axes[i].tick_params(axis='x', rotation=90)
                axes[i].set_title(f'{col} vs {y}')
            # Caso 3: Fecha vs Numérico -> Lineplot
            elif col in date_cols and y in data.select_dtypes("number"):
                date_data = data[col].dt.year if dateplot == "year" else data[col].dt.month
                sns.lineplot(x=date_data, y=data[y], ax=axes[i])
                axes[i].set_title(f'{col}_{dateplot} vs {y}')
                axes[i].tick_params(axis='x', rotation=90)
            else:
                print(f"Advertencia: No se pudo graficar '{col}' frente a '{y}' ya que no es de un tipo soportado o la combinación no es válida.")

            axes[i].set_xlabel("")
        
        for j in range(len(x), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
