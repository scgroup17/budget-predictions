# Budget Predictions ML Service - Estructura Modular

## Estructura de Carpetas

```
app/
├── __init__.py
├── app.py                          # Aplicación principal Flask
├── config.py                       # Configuración y constantes globales
├── api/
│   ├── __init__.py
│   ├── schemas.py                  # Esquemas Swagger/OpenAPI
│   ├── routes.py                   # Endpoints principales (predict, retrain, health)
│   └── categories_route.py         # Endpoint de categorías
├── models/
│   ├── __init__.py
│   └── model_loader.py             # Carga de modelos desde Supabase
└── services/
    ├── __init__.py
    ├── prediction_service.py       # Lógica de predicción
    └── retraining_service.py       # Lógica de reentrenamiento

run.py                              # Punto de entrada
```

## Módulos

### `config.py`
- Variables de entorno y configuración
- Estado global de modelos (MODELS, LABEL_ENCODERS, MODEL_PERFORMANCE)
- Logger configurado

### `models/model_loader.py`
- `load_models_from_storage()`: Carga modelos desde Supabase Storage
- `get_model_tier()`: Calcula tier del modelo (A/B/C)

### `services/prediction_service.py`
- `encode_features()`: Codifica features categóricas
- `build_feature_dict()`: Construye diccionario de features
- `predict_budget_item()`: Predicción principal

### `services/retraining_service.py`
- `fetch_training_data()`: Obtiene datos de Supabase
- `check_enrichment()`: Verifica enriquecimiento de propiedades
- `prepare_training_dataset()`: Prepara dataset de entrenamiento
- `encode_training_features()`: Codifica features para entrenamiento
- `train_category_models()`: Entrena modelos por categoría
- `save_models_to_storage()`: Guarda modelos en Supabase
- `log_retraining()`: Registra resultados de reentrenamiento
- `retrain_models()`: Orquestación principal del reentrenamiento

### `api/schemas.py`
- `register_schemas()`: Registra todos los esquemas Swagger/OpenAPI

### `api/routes.py`
- `create_namespaces()`: Crea namespaces de API
- `register_routes()`: Registra endpoints de predict, retrain, health

### `api/categories_route.py`
- `register_categories_route()`: Registra endpoint de categorías

### `app.py`
- `create_app()`: Factory para crear aplicación Flask
- `main()`: Punto de entrada principal

## Uso

### Ejecutar el servicio:
```bash
python run.py
```

### Ejecutar con el archivo original (sin cambios):
```bash
python ml_flask_service.py
```

Ambos métodos funcionan de la misma manera. La estructura modular facilita:
- Mantenimiento del código
- Testing unitario
- Reutilización de componentes
- Mejor organización y legibilidad
