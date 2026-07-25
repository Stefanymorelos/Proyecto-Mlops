# MLOps Pipeline

Proyecto de pipeline de Machine Learning con estructura estandarizada para despliegue automatizado vía Jenkins.

## Estructura del proyecto

```
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb
│       ├── comprension_eda.ipynb
│       ├── ft_engineering.py
│       ├── model_training_evaluation.py
│       ├── model_deploy.py
│       └── model_monitoring.py
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── setup.bat
├── config
└── readme.md
```

## Ramas

- `master`: código estable en producción
- `certification`: código en fase de certificación/QA
- `developer`: desarrollo activo

## Setup

Ejecutar `setup.bat` para crear el entorno virtual e instalar dependencias.

## Autor

Stefany
