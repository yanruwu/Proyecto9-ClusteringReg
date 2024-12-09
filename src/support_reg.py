# -------------------- VISUALIZACIONES --------------------
# ---------------------------------------------------------------
# Librerías utilizadas para crear gráficos, visualizar datos y resultados
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- MODELOS Y REGRESIÓN --------------------
# ---------------------------------------------------------------
# Librerías para crear y ajustar los modelos de regresión
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

# -------------------- PREPROCESAMIENTO Y SELECCIÓN DE DATOS --------------------
# ---------------------------------------------------------------
# Funciones necesarias para dividir los datos en entrenamiento y prueba,
# y realizar la búsqueda de hiperparámetros con GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV

# -------------------- MÉTRICAS --------------------
# ---------------------------------------------------------------
# Funciones para calcular las métricas de rendimiento de los modelos
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# -------------------- OTRAS --------------------
# ---------------------------------------------------------------
# Librerías adicionales que podrían usarse para procesamiento y análisis de datos
import numpy as np
import pandas as pd

import shap



def create_model(params, X_train, y_train, method = DecisionTreeRegressor(), cv= 5, scoring = "neg_mean_squared_error"):
    """
    Crea y ajusta un modelo utilizando búsqueda en cuadrícula para encontrar los mejores hiperparámetros.

    Parameters:
        params (dict): Diccionario de parámetros a probar en la búsqueda en cuadrícula.
        X_train (array-like): Conjunto de características de entrenamiento.
        y_train (array-like): Etiquetas del conjunto de entrenamiento.
        method (estimator, optional): Modelo de regresión a usar (por defecto DecisionTreeRegressor).
        cv (int, optional): Número de pliegues en la validación cruzada (por defecto 5).
        scoring (str, optional): Métrica de evaluación a usar en la búsqueda en cuadrícula (por defecto "neg_mean_squared_error").

    Returns:
        grid_search (GridSearchCV): Objeto de GridSearchCV ajustado con los mejores parámetros encontrados.
    """
    grid_search = GridSearchCV(estimator = method, param_grid=params, cv = cv, scoring = scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search

def metricas(y_train, y_train_pred, y_test, y_test_pred):
    """
    Calcula métricas de regresión para conjuntos de entrenamiento y prueba.
    
    Parameters:
        y_train (array-like): Valores reales del conjunto de entrenamiento.
        y_train_pred (array-like): Predicciones del conjunto de entrenamiento.
        y_test (array-like): Valores reales del conjunto de prueba.
        y_test_pred (array-like): Predicciones del conjunto de prueba.
    
    Returns:
        dict: Diccionario con métricas de R², MAE, MSE y RMSE.
    """
    metricas = {
        'train': {
            'r2_score': r2_score(y_train, y_train_pred),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2_score': r2_score(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
    }
    return metricas


class RegressionModel:
    """
    Clase para crear, entrenar y evaluar modelos de regresión.
    
    Attributes:
        X_train (array-like): Conjunto de características de entrenamiento.
        X_test (array-like): Conjunto de características de prueba.
        y_train (array-like): Etiquetas del conjunto de entrenamiento.
        y_test (array-like): Etiquetas del conjunto de prueba.
        model (estimator): Modelo de regresión entrenado.
        metrics_df (DataFrame, optional): DataFrame con las métricas de evaluación.
        best_params (dict, optional): Los mejores parámetros encontrados durante la búsqueda de hiperparámetros.
    """
    def __init__(self, X, y, test_size=0.3, random_state=42):
        """
        Inicializa el modelo de regresión y divide los datos en entrenamiento y prueba.

        Parameters:
            X (array-like): Características del conjunto de datos.
            y (array-like): Etiquetas del conjunto de datos.
            test_size (float, optional): Fracción de los datos a utilizar para el conjunto de prueba (por defecto 0.3).
            random_state (int, optional): Semilla para la aleatoriedad en la división de datos (por defecto 42).
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model = None
        self.metrics_df = None
        self.best_params = None
        self.random_state = random_state
        self.results = {}
    
    def _get_model(self, model_type, learning_rate=0.1):
        """
        Obtiene el modelo seleccionado según el tipo indicado.

        Parameters:
            model_type (str): Tipo de modelo a usar ("linear", "decision_tree", "random_forest", "gradient_boosting").
            learning_rate (float, optional): Tasa de aprendizaje para el modelo de GradientBoosting (por defecto 0.1).

        Returns:
            estimator: Modelo de regresión correspondiente al tipo seleccionado.

        Raises:
            ValueError: Si el tipo de modelo no es válido.
        """
        models = {
            "linear": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=self.random_state),
            "random_forest": RandomForestRegressor(random_state=self.random_state),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.random_state, learning_rate=learning_rate),
            "xgboost": XGBRegressor(random_state=self.random_state, learning_rate=learning_rate, verbosity=1)
        }
        if model_type not in models:
            raise ValueError(f"El modelo '{model_type}' no es válido. Elija uno de {list(models.keys())}")
        return models[model_type]

    def train(self, model_type, params=None, learning_rate=0.1):
        """
        Entrena el modelo seleccionado con los datos de entrenamiento y calcula las métricas de evaluación.

        Parameters:
            model_type (str): Tipo de modelo a usar ("linear", "decision_tree", "random_forest", "gradient_boosting").
            params (dict, optional): Parámetros para la búsqueda en cuadrícula (por defecto None).
            learning_rate (float, optional): Tasa de aprendizaje para el modelo de GradientBoosting (por defecto 0.1).

        Returns:
            DataFrame: DataFrame con las métricas de evaluación para los conjuntos de entrenamiento y prueba.
        """
        self.model = self._get_model(model_type, learning_rate)
        
        if params:
            grid_search = GridSearchCV(self.model, param_grid=params, cv=5, scoring="r2", n_jobs=-1, verbose = 1)
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        
        else:
            self.model.fit(self.X_train, self.y_train)
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        self.metrics_df = pd.DataFrame({
            "Train": [
                r2_score(self.y_train, y_train_pred),
                mean_absolute_error(self.y_train, y_train_pred),
                root_mean_squared_error(self.y_train, y_train_pred)
            ],
            "Test": [
                r2_score(self.y_test, y_test_pred),
                mean_absolute_error(self.y_test, y_test_pred),
                root_mean_squared_error(self.y_test, y_test_pred)
            ]
        }, index=["R2", "MAE", "RMSE"]).T

        self.results[model_type] = {
            "metrics": self.metrics_df,
            "best_model": self.model
        }
        
        return self.metrics_df

    def display_metrics(self):
        """
        Muestra las métricas de evaluación del modelo.

        Si las métricas no están disponibles, muestra un mensaje indicándolo.
        """
        if self.metrics_df is not None:
            display(self.metrics_df)
        else:
            print("No hay métricas disponibles. Primero entrena el modelo.")
    
    def plot_residuals(self):
        """
        Muestra los gráficos de residuos para los conjuntos de entrenamiento y prueba.

        Si el modelo no ha sido entrenado previamente, muestra un mensaje indicándolo.
        """
        if self.model is None:
            print("Primero debes entrenar un modelo para graficar los residuos.")
            return
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        plt.figure(figsize=(12, 6))

        # Residuos de entrenamiento
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=self.y_train, y=y_train_pred, color="blue", alpha=0.6)
        plt.plot([min(self.y_train), max(self.y_train)], [min(y_train_pred), max(y_train_pred)], color="red", ls="--")
        plt.title("Train")
        plt.xlabel("Valores Reales")
        plt.ylabel("Valores predichos")

        # Residuos de prueba
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=self.y_test, y=y_test_pred, color="green", alpha=0.6)
        plt.plot([min(self.y_test), max(self.y_test)], [min(y_test_pred), max(y_test_pred)], color="red", ls="--")
        plt.title("Test")
        plt.xlabel("Valores Reales")
        plt.ylabel("Valores predichos")

        plt.tight_layout()
        plt.show()
    
    def get_best_params(self):
        """
        Obtiene los mejores parámetros del modelo si se ha realizado una búsqueda en cuadrícula.

        Esta función devuelve los parámetros óptimos encontrados durante una búsqueda en cuadrícula,
        si dicha búsqueda se ha realizado previamente. Si no se ha realizado ninguna búsqueda en cuadrícula
        o no hay parámetros disponibles, se muestra un mensaje y se retorna `None`.

        Returns:
            dict or None: Diccionario con los mejores parámetros si se realizó la búsqueda en cuadrícula, 
                        o `None` si no hay parámetros disponibles.
        """
        # Obtener los mejores parámetros si se realizaron búsquedas en cuadrícula
        if self.best_params:
            return self.best_params
        else:
            print("No se ha realizado búsqueda en cuadrícula o no hay parámetros disponibles.")
            return None

    def return_model(self):
        """
        Retorna el modelo actual.

        Esta función devuelve el modelo entrenado o el modelo base utilizado en la instancia. 
        Es útil para obtener el modelo que se ha entrenado o ajustado y utilizarlo para predicciones o evaluaciones posteriores.

        Returns:
            estimator: El modelo entrenado o el modelo base usado en la instancia.
        """
        return self.model
    

def shap_summary(model, X_train, X_test, plot_type="bee_swarm"):
    """
    Genera un SHAP explainer y un summary plot para un modelo de regresión.

    Parameters:
        model: Modelo entrenado.
        X_train: Datos de entrenamiento (usados para inicializar el explainer).
        X_test: Datos de prueba (usados para calcular los valores SHAP).
        plot_type: Tipo de gráfico de resumen ("bee_swarm" o "bar").
    
    Returns:
        explainer: El objeto SHAP explainer creado.
        shap_values: Los valores SHAP calculados para X_test.
    """
    # Crear el explainer de SHAP
    explainer = shap.Explainer(model, X_train)
    
    # Calcular los valores SHAP
    shap_values = explainer(X_test, check_additivity=False)
    
    # Generar el gráfico
    plt.figure()
    if plot_type == "bee_swarm":
        shap.summary_plot(shap_values, X_test)  # Gráfico tipo bee swarm
    elif plot_type == "bar":
        shap.summary_plot(shap_values, X_test, plot_type="bar")  # Gráfico tipo barras
    else:
        raise ValueError("El plot_type debe ser 'bee_swarm' o 'bar'")
    
    return explainer

