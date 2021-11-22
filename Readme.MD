# Readme

```bash
# Start database and worker services
docker-compose up -d

# Start API
uvicorn mltest.app:app --reload
```

## Framework

- Wie sehen die Metadaten aus?

- Framework
    - Auf welchen Plattformen kann das Framework ausgeführt werden? (SKlearn -> CPU, Tensorflow -> [CPU,GPU])
    - was ist mit ARM? FPGA?

- Task (Klassifikation, Regression, Generative, Clustering)

- Tests auf verschiedener Hardware?? Einfach Kreuzprodukt über Plattformen, auf denen Framework kompatibel ist?

- Welcher Test ist auf welcher Hardware ausführbar? NVIDIA-Messung nur auf Nivdia GPU möglich.

- Was haben wir für Tests?
    - Resourcen [Laufzeit, Energie, Speicher] (Tests werden X mal ausgeführt)
        - Limitieren wir die Daten für die Tests? e.g. teste nur auf 10.000 Samples
        - Laufzeit:
            - Für NN verschiedene Batchsizes [1, 16, 32, 64, 128]
            - Für RandomForest: Different Exection engine (ONNX/FastInference)
    - Energie (geht aktuell nur für Intel-CPUs und Nvidia-GPU)
    - Speicher:
        - Modellgröße
            - Binarysize? Modellparameter?
        - Speicher, der während der Ausführung benötigt wird
        - GPU-Speicher

    - Modelleigenschaften
        - Robustness
            - Für NN: [RobustBench](https://github.com/RobustBench/robustbench)
            - Für XGB: https://github.com/laudv/veritas ??
            - Für DT: ???
            - SVM: https://github.com/abstract-machine-learning/saver

    - Metriken:
        - Accuracy, Precision, Recall, F1, AUC, ROC, Confusion Matrix, ...
        - MSE, MAE, ...

- Wie wird quantisierung behandelt?
    - Modell wird submitted -> Model -> [Model8Bit]
    - Execute tests for each model..

###         

```python

# Probably hardware indepenedent
metrics = [{
    'name': 'accuracy',
    'task': ['classification'],
    'module': '...',
    'clz': '...'
}]

robustness_tests = [{
    '???': '???',
    'frameworks_allowed': ['...'],
    '??task???': '???'
}]

resource_tests = [{
    'frameworks_allowed': ['...'],  # FastInference only for trees
    'hardware_allowed': ['...']
}]

```

## Was bräuchte man an Metadaten?

- Task (classification)
- Framework
- ModelTyp


- Dataset Properties
    - InputShape
    - Anzahl Klassen
    - Anzahl Samples
    - Class distribution (fairness?)

### ModelTyp: NeuronalesNetz

-

```bash
config = {
  'general_meta1': '...',
  'model_meta': [NeuralNetwork|RandomForest]
}

NeuralNetwork = ?
RandomForest = ?

general_meta = ?

```

### TestRegistry

- Frage1: Wie speichern wir Tests ab?
- Frage2: Wie / Welche Metadaten speichern wir?

- Frage3: Wie matchen wir das?

#### Frage1:

- Gruppen von Tests? e.g. Resourcen oder Laufzeit? -> Gemeinsame Interfaces
    - Welche Tests sind wie allgemein? Seperate Tests für NN und DT?`Welche gehen für beide?
- Speicher Klasse und Modul, Instantiierung zur Laufzeit
- Benötige Parameter??

- Wie sieht eine TestConfiguration?
    - Liste/Dict von Tests

```bash
   [
      'test1': {
        'module': '...',
        'clz': '...',
        'params': {
          'param1': '...',
          'param2': '...'
        }
      },
   ]
```

### Offene Fragen

- Was wird für XAI, Fariness, Bias benötigt?
- Wenn wir Modelle in andere Formate konvertieren (e.g. ONNX/FI), reichen uns die Resourcen oder benötigen wir auch die Metriken?

### Further frameworks

- https://github.com/microsoft/hummingbird
