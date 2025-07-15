# Extracción de Conceptos Médicos

Este repositorio contiene el código utilizado para la creación de una herramienta de extracción de información que permite el procesamiento de historias clínicas de cáncer de pulmón de manera automática utilizando conceptos de Fine-Tuning y Transfer-Learning e integrando el modelo __, el cual fue pre-entrenado en lenguaje natural con un corpus anotado para extracción de entidades nombradas.

## Archivos

- `entregable_1.py`: script donde se muestra el proceso de afinamiento del corpus de cáncer de pulmón otorgado.
- `entregable_2.py`: script donde se muestra el proceso de validación del modelo anterior.


## Modelos utilizados
- [`roberta-base-biomedical-clinical-es`](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)

## Requisitos

```bash
numpy
pandas
datasets
collections
pathlib
transformers
