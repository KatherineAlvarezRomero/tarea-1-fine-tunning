# Extracción de Conceptos Médicos

Este repositorio contiene el código utilizado para la creación de una herramienta de extracción de información que permite el procesamiento de historias clínicas de cáncer de pulmón de manera automática utilizando conceptos de Fine-Tuning y Transfer-Learning e integrando el modelo __, el cual fue pre-entrenado en lenguaje natural con un corpus anotado para extracción de entidades nombradas.

## Archivos

- `entregable_1.py`: script que utiliza un modelo NER preentrenado y ajustado con un dataset de cáncer de mama en español para extraer entidades de las historias clínicas cáncer de mama.
- `e1_entidades_extraidas_token.xlsx`: resultados con las columnas `id_historia`, `oracion_id`, `oraciones`, `Palabra`, `Entidad` y `Score`.
- `e1_entidades_extraidas_reconstruidas.xlsx`: resultados con tokens reconstruídos con las columnas `id_historia`, `oracion_id`, `oraciones`, `Palabra` y `Entidad`.
- `entregable_2.py`: script que utiliza un modelo de detección de negación/incertidumbre para detectar negación e incertidumbre en las historias clínicas cáncer de mama.
- `e2_entidades_extraidas_negacion_token.xlsx`: resultados con las columnas `id_historia`, `texto_original`, `entidad`, `etiqueta` y `score`.
- `e2_entidades_extraidas_negacion_clasificado.xlsx`: resultados con tokens reconstruídos y etiquetas clasificadas con las columnas `id_historia`, `texto_original`, `entidad`, `etiqueta`, `estado` y `score`.
- `entregable_3.py`: script que integra el modelo NER y el modelo de detección de negación/incertidumbre para detectar negación e incertidumbre en las entidades extraídas de las historias clínicas cáncer de mama.
- `e3_base_estructurada_final.xlsx`: resultados con las columnas `id_historia`, `oracion`, `NER` y `Estado`.

## Modelos utilizados
- NER: [`anvorja/breast-cancer-biomedical-ner-sp`](https://huggingface.co/anvorja/breast-cancer-biomedical-ner-sp)
- Negación/Incertidumbre: [`JuanSolarte99/bert-base-uncased-finetuned-ner-negation_detection_NUBES`](https://huggingface.co/JuanSolarte99/bert-base-uncased-finetuned-ner-negation_detection_NUBES)

## Requisitos

```bash
transformers
torch
pandas
tqdm
