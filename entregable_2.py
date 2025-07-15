from entregable_1 import train_model

def test_metrics(trainer, tokenized_datasets):
    """
    Evalúa el rendimiento del modelo entrenado sobre el conjunto de prueba.

    Esta función utiliza el método `evaluate` del objeto `trainer` para calcular 
    las métricas de evaluación sobre el dataset de prueba tokenizado. 
    Luego, imprime las métricas obtenidas en consola.

    Requiere que `trainer` y `tokenized_datasets["test"]` estén previamente definidos.
    """
    metrics = trainer.evaluate(tokenized_datasets["test"])
    print("Test metrics:")
    print(metrics)

def save_model(name_save, trainer, tokenizer):
    """
    Guarda el modelo entrenado, el tokenizer y su configuración en una carpeta específica.

    Parámetros:
    ----------
    name_save : str
        Ruta o nombre de la carpeta donde se guardará el modelo y el tokenizer.

    Esta función utiliza el método `save_model` del objeto `trainer` y 
    el método `save_pretrained` del `tokenizer` para almacenar los recursos.
    """
    trainer.save_model(name_save)
    tokenizer.save_pretrained(name_save)

def test_model(trainer, model, tokenizer):
    """
    Realiza inferencia de entidades nombradas (NER) sobre un conjunto de oraciones clínicas de ejemplo.

    Esta función crea un pipeline de NER utilizando el modelo y tokenizer previamente entrenados,
    y aplica dicho pipeline a un conjunto de oraciones relacionadas con el cáncer de pulmón.

    Para cada oración, se imprimen las entidades detectadas junto con su tipo y score de confianza.

    Requiere que las variables `model` y `tokenizer` estén previamente cargadas.
    """
    oraciones_ejemplo = [
        "El paciente presenta carcinoma pulmonar en estadio avanzado.",
        "Se observó masa en lóbulo superior derecho compatible con tumor.",
        "No se evidencia metástasis en TAC torácico.",
        "Paciente con antecedentes de tabaquismo por 20 años.",
        "Se inicia tratamiento con quimioterapia basada en platino.",
        "La biopsia confirmó adenocarcinoma de pulmón.",
        "El informe histopatológico reveló carcinoma de células no pequeñas.",
        "La lesión mide 3.2 cm y compromete bronquios proximales.",
        "Se indicó radioterapia como terapia adyuvante.",
        "Paciente inició esquema de cisplatino y pemetrexed con buena tolerancia."
    ]

    from transformers import pipeline

    # Crear el pipeline de NER con el modelo ya entrenado
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    # Procesar cada oración y mostrar resultados
    for i, oracion in enumerate(oraciones_ejemplo, 1):
        print(f"\nOración {i}: {oracion}")
        predicciones = ner_pipeline(oracion)
        for entidad in predicciones:
            print(f"  - {entidad['word']} ({entidad['entity_group']}): score={entidad['score']:.2f}")



if __name__ == "__main__":
    """
    Función principal con menú interactivo para ejecutar pruebas sobre el modelo entrenado.
    
    Opciones disponibles:
    1. Evaluar métricas del conjunto de prueba.
    2. Guardar el modelo y el tokenizer.
    3. Probar el modelo con oraciones de ejemplo (NER).
    0. Salir del programa.
    """

    model, trainer, tokenizer, model_name, tokenized_datasets = train_model()
    
    print("=== INICIANDO PROGRAMA ===")
    print(f"Modelo a usar: {model_name}")

    while True:
        print("\n=== MENÚ DE OPCIONES ===")
        print("1. Evaluar métricas en el conjunto de prueba")
        print("2. Guardar modelo y tokenizer")
        print("3. Probar el modelo con oraciones de ejemplo (NER)")
        print("0. Salir")
        
        opcion = input("Seleccione una opción (0-3): ").strip()
        
        if opcion == "1":
            print("\n[Evaluando métricas del conjunto de prueba...]")
            test_metrics(trainer, tokenized_datasets)
        elif opcion == "2":
            nombre = input("Ingrese el nombre o ruta donde desea guardar el modelo: ").strip()
            print(f"\n[Guardando modelo en '{nombre}'...]")
            save_model(nombre, trainer, tokenizer)
        elif opcion == "3":
            print("\n[Probando modelo con oraciones de ejemplo...]")
            test_model(trainer, model, tokenizer)
        elif opcion == "0":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente nuevamente.")
