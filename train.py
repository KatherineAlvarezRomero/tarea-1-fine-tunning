import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset, Features, Sequence, Value, ClassLabel
import evaluate
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

#Función para cargar los dataset
def cargar_csv(path):
    return pd.read_csv(path)

#Función para limpiar los dataset
def limpiar_dataset(df):
    return df.dropna(subset=["Word", "Tag"])

#Agrupar oraciones
def agrupar_oraciones(df):
    oraciones = []
    oracion_actual = []
    for i, fila in df.iterrows():
        if pd.notna(fila['Sentence #']):
            if oracion_actual:
                oraciones.append(oracion_actual)
            oracion_actual = [(fila['Word'], fila['Tag'])]
        else:
            oracion_actual.append((fila['Word'], fila['Tag']))
    if oracion_actual:
        oraciones.append(oracion_actual)
    return oraciones

#Convertir oraciones agrupadas a formato compatible con Hugging Face Dataset
def dataset_formato(oraciones):
    tokens_list = []
    tags_list = []

    for i in oraciones:
        tokens = []
        tags = []
        for word, tag in i:
            tokens.append(word)
            tags.append(tag)
        tokens_list.append(tokens)
        tags_list.append(tags)

    return {"tokens": tokens_list, "ner_tags": tags_list}

#####################################################################################
ruta_archivos = "."
global model_checkpoint
model_checkpoint = "./roberta-base-biomedical-clinical-es"

train_df = cargar_csv(f"{ruta_archivos}/sentences_train.csv")
dev_df   = cargar_csv(f"{ruta_archivos}/sentences_dev.csv")
test_df  = cargar_csv(f"{ruta_archivos}/sentences_test.csv")
######################################################################################

#Validación de datos NaN
print(train_df.isnull().sum())
print(dev_df.isnull().sum())
print(test_df.isnull().sum())

train_df = limpiar_dataset(train_df)
dev_df   = limpiar_dataset(dev_df)
test_df  = limpiar_dataset(test_df)

train_oraciones = agrupar_oraciones(train_df)
dev_oraciones = agrupar_oraciones(dev_df)
test_oraciones = agrupar_oraciones(test_df)

train_data = dataset_formato(train_oraciones)
dev_data = dataset_formato(dev_oraciones)
test_data = dataset_formato(test_oraciones)

# Detectar automáticamente todas las etiquetas del conjunto de entrenamiento
LABELS = sorted(set(tag for seq in train_data["ner_tags"] for tag in seq if isinstance(tag, str)))

# Definir la estructura de features con las etiquetas detectadas
features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=LABELS))
})

# Convertir datos a formato DatasetDict compatible
dataset = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(dev_data),
    "test": Dataset.from_dict(test_data)
})

dataset = dataset.cast(features)

# Tokenización y alineación
global tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(_label):
    tokenized_inputs = tokenizer(_label["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(_label["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

global tokenized_datasets
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Mapeo explícito de etiquetas
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

#Configuración de entrenamiento
global model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir=f"./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none"
)

label_feature = dataset["train"].features["ner_tags"].feature  # ← accede al ClassLabel dentro de Sequence
id2tag = {i: label_feature.int2str(i) for i in range(len(label_feature.names))}

#Métricas
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#Entrenamiento
def train_model():
    """
    Entrena el modelo con los datos tokenizados y configuraciones definidas.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return model, trainer, tokenizer, model_checkpoint, tokenized_datasets