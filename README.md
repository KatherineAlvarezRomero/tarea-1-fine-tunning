# Extracción de Conceptos Médicos

Este repositorio contiene el código utilizado para la creación de una herramienta de extracción de información que permite el procesamiento de historias clínicas de cáncer de pulmón de manera automática utilizando conceptos de Fine-Tuning y Transfer-Learning e integrando el modelo __, el cual fue pre-entrenado en lenguaje natural con un corpus anotado para extracción de entidades nombradas.

## Archivos

- `entregable_1.py`: script donde se muestra el proceso de afinamiento del corpus de cáncer de pulmón otorgado.
- `entregable_2.py`: script donde se muestra el proceso de validación del modelo anterior.


## Modelos utilizados
- [`roberta-base-biomedical-clinical-es`](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)

`git clone https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`

## Requisitos

```bash
accelerate==1.8.1
aiohappyeyeballs==2.6.1
aiohttp==3.12.13
aiosignal==1.4.0
attrs==25.3.0
certifi==2025.6.15
charset-normalizer==3.4.2
colorama==0.4.6
datasets==3.6.0
dill==0.3.6
evaluate==0.4.4
filelock==3.18.0
frozenlist==1.7.0
fsspec==2025.3.0
huggingface-hub==0.33.2
idna==3.10
Jinja2==3.1.6
joblib==1.5.1
MarkupSafe==3.0.2
mpmath==1.3.0
multidict==6.6.3
multiprocess==0.70.14
networkx==3.5
numpy==2.3.1
packaging==25.0
pandas==2.3.0
propcache==0.3.2
psutil==7.0.0
pyarrow==20.0.0
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.4
responses==0.18.0
safetensors==0.5.3
scikit-learn==1.7.0
scipy==1.16.0
seqeval==1.2.2
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.21.2
torch==2.7.1
tqdm==4.67.1
transformers==4.53.1
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
xxhash==3.5.0
yarl==1.20.1
```
