# Actividad Semana 4: Código de experimentación base con MLFlow

## Cómo correr este proyecto
### Paso 1: Instalar dependencias
Asegúrate de tener instalado Python 3.8 o una versión superior. A continuación, crea un entorno virtual e instala las dependencias necesarias:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Paso 2: Iniciar el servidor MLFlow
Inicia el servidor de MLFlow con el siguiente comando:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### Paso 3: Correr script de python para registrar un experimento

Por ejemplo:

```bash
python train.py \
	--data-path "bank_marketing.csv" \
	--target "target" \
	--model-type "logistic_regression" \
	--C 1.0 \
	--l1-ratio 0 \
	--class-weight "balanced" \
	--max-iter 1000 \
	--experiment-name "bank-marketing-predictor" \
	--run-name "lr-c1-l2-balanced-mi1000"
```

Esto generará una ejecución en MLFlow con los parámetros y métricas especificados. Accede a `http://127.0.0.1:5000` en tu navegador web para ver la interfaz de usuario de MLFlow y explorar las ejecuciones registradas.
Cada ejecución está configurada para registrar los siguientes argumentos, parámetros y métricas.
Datos:
- `data-path`: Ruta en el directorio local donde se encuentra el dataset
- `target`: Nombre de la columna donde se encuentran las etiquetas.
- `model-type`: Modelo a utilizar, `"logistic_regression"` por default.

Parámetros:
- `C`: Intensidad de la regularización
- `l1-ratio`: El parámetro de mezcla de Elastic-Net
- `class-weight`: Pesos asociados a las clases
- `max-iter`: Número máximo de iteraciones para el solucionador

Métricas:
- `accuracy`: Precisión del modelo en el conjunto de prueba
- `f1_score`: Puntuación F1 del modelo en el conjunto de prueba
- `recall_yes`: Exhaustividad para la clase positiva en el conjunto de prueba
- `roc_auc`: Área bajo la curva ROC en el conjunto de prueba

**Nota**: Por defecto, se usa regresión logística de sklearn, con el hiperparámetro `solver="saga""`, en una siguiente versión, podríamos actualizar el tipo de solucionador, así como el modelo con sus respectivos hiperpárametros y métricas.

Además, cada vez que ejecutes el experimento, se generará un gráfico de matriz de confusión y se registrará como un artefacto en MLFlow. Puedes ver este gráfico en la interfaz de usuario de MLFlow, en la sección "Artifacts" de la ejecución correspondiente.

Por último, el modelo se registra en el Registro de modelos de MLFlow, lo que te permite administrar e implementar diferentes versiones de tu modelo con facilidad. Puedes ver los modelos registrados en la interfaz de usuario de MLFlow, en la sección "Model Registry".