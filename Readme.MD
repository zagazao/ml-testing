# Readme

```bash
# Create env...

set -o allexport
source dev.env 
set +o allexport

# Start database and worker services
docker-compose up --build -d

# Start API
uvicorn mltest.app:app --reload

cd mltest 
python3 my_client.py
```

## Tools

Im Folgenden werden einige Tools genannt, welche zur Realsisierung des Frameworks genutzt werden können. 
[Docker Compose](https://docs.docker.com/compose/) ist ein sehr hilfreiches Tool, welches es ermöglicht, mehrere Container
gleichzeitig zu orchestrieren. Im der aktuellen `docker-compose.yml` werden die folgenden Container/Services definiert:

### MongoDB

[MongoDB](https://www.mongodb.com/) ist eine nicht relationale Datenbank und bietet eine sehr gute Unterstützung für JSON-artige Dokumente. 
In der Datenbank könnten die Metadaten genauso wie die Ergebnisse der Experimente gespeichert werden. Mit Hilfe der
[Mongo Query Language](https://docs.mongodb.com/manual/tutorial/query-documents/) können die Ergebnisse / Metadaten 
abgefragt und aggreggiert werden.

In der aktuellen Konfiguration läuft die MongoDB auf Port `27017`. Nutzername und Passwort können über die Umbegunbsvariablen
`MONGO_INITDB_ROOT_USERNAME` und `MONGO_INITDB_ROOT_PASSWORD` gesetzt werden. Die Zugangsdaten lauten wie folgt:
`secret_mongo_user:secret_mongo_password`. Aktuell werden die Daten nicht [persistent gespeichert](https://hub.docker.com/_/mongo). Um dies zu ermöglichen, müsste ein Verzeichnis
oder ein [Docker-Volume](https://docs.docker.com/storage/volumes/) nach `/data/db` gemountet werden.


### S3 Object Storage

In einem [Object storage](https://en.wikipedia.org/wiki/Object_storage) können beliebige binäre Daten / Dateien gespeichert werden. Somit kann dieser Service zur Sicherung von 
Modellen und Datensätzen verwendet werden. Als Alternative zu bezahlten Diensten, wie beispielsweise AWS S3, bietet sich der 
selbstbetriebene [Minio S3 Object-Storage an](https://min.io/) an. In der aktuellen Konfiguration wird der Service auf Port `9000` gestartet, die Managementkonsole läuft auf Port `9001`.
Die Zugangsdaten können, wie bei der MongoDB, über die Umbegunbsvariablen `MINIO_ROOT_USER` und `MINIO_ROOT_PASSWORD` gesetzt werden. Aktuell lauten sie `storage_user:crazy_storage_password`.

Aus einem Pythonskript kann über den [Minio-Client](https://docs.min.io/docs/python-client-api-reference.html) auf den Storage zugegriffen werden. [Hier](mltest/clients.py) ist ein
Beispiel, wie der Client erstellt wird. Über die Funktionen [put_object](https://docs.min.io/docs/python-client-api-reference.html#put_object) und
[get_object](https://docs.min.io/docs/python-client-api-reference.html#get_object) können die Daten gelesen und gespeichert werden.
Modelle oder Datensätze können z.B. in [io.BytesIo](https://docs.python.org/3/library/io.html#io.BytesIO) Objekten durch Pickle oder 
[np.savez](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) Aufrufe gespeichert und anschließend serialisiert werden. 

Neben dem eigentlichen Service-Container gibt es den `createbuckets`-Service , welcher den benötigten Bucket erstellt.

### Queue

Um die verschiedenen Queues zu realisieren, bietet sich das Python-Framework [Celery](https://docs.celeryproject.org/en/stable/) an. Das Framework arbeitet
auf dem Messagebroker [RabbitMQ](https://www.rabbitmq.com/) und nimmt dem Nutzer die Queue-Konfiguration und Serialisierung ab.

Dazu werden vom Nutzer [Tasks](https://docs.celeryproject.org/en/stable/userguide/tasks.html) als Pythonfunktion oder -klasse definiert und im Framework
[registriert](https://stackoverflow.com/a/41794225). Anschließend können die Queues und deren Routing
[konfiguriert](https://docs.celeryproject.org/en/stable/userguide/routing.html) werden. Hier können bestimmte Tasks auf spezifische Queues gemappt werden. Worker können so 
gestartet werden, dass sie nur Jobs aus spezifischen Queues abarbeiten. Um einen Worker so zu starten, dass er z.B. nur Tasks aus `queue1` und `queue2` ababarbeitet, kann er mit
folgender Kommandozeile aufgerufen werden:
```bash
celery -A proj worker -Q feeds,celery
```
Der Jobstatus sowie das Ergebnis werden in einem Resultbackend gespeichert. Standardmäßig wird [Redis](https://redis.io/) verwendet, welches ebenfalls hier adaptiert wird.

Ebenfalls wird der [Flower](https://flower.readthedocs.io/en/latest/) -Service gestartet. Bei Flower handelt es sich um ein Monitoring-Tool, welches Informationen über die Celery-Worker und -Tasks bereitstellt. Dieser Server läuft auf Port `5555`. 

Ingesamt werden die folgenden Komponenten auf den genannten Ports für das Queuesystem in der `docker-compose.yml` gestartet:
```bash

RabbitMQ 5672
Redis 6379
CeleryWorker
Celery-Monitoring (Flower) 5555
```

### Backend / API

Zur Realisierung eines Rest-Backends kann [FastAPI](https://fastapi.tiangolo.com/) verwendet werden. Dieses Framework ermöglicht es, eine REST-Schnittstelle zu realisieren, über 
die beispielsweise Tasks submitted werden können oder Ergebnisse abgefragt werden können. Aktuell läuft diese auf dem Host, um 
das Live-Reload-Feature auszunutzen. 

Um die API auf Port `8000` zu starten, kann folgendes Kommando ausgeführt werden:
```bash
uvicorn mltest.app:app --reload
```
Die API ist [OpenAPI](https://www.openapis.org/) konform und stellt Docs sowie Testmöglichkeiten unter `localhost:8000/docs` bereit. 

Zur Kommunikation aus Python ist es empfehlenswert einen Client zu schreiben, welcher Pythonfunktionen auf die entsprechenden Endpunkte mappt. Ein prototypischer 
Client ist in [mltest/my_client.py](mltest/my_client.py) zu finden.

Das restliche Repository besteht aus Laufzeittests für Zufallswälder. Die Idee ist es, ein trainiertes Modell sowie einen Validierungsdatensatz hochzuladen, woraufhin dieses Modell in
verschiedenen Frameworks mit unterschiedlichen Optimierungsstufen auf seine Laufzeit zu testen. Folgende Frameworks werden aktuell getestet:
- Sklearn
- Sklearn 2 Onnx
- FastInference mit verschiedenen Optimierungen
- Treelite


#### Datensätze

Zum Laden und Speichern von Datensätzen könnte sich ein hybrides S3/Mongo-Konzept, inspiriert von MLFlow, eignen. Die Metadaten würden mit einem Verweis auf den Bucket und 
Dateinamen in einer MongoDB -Collection gespeichert. Die eigentlichen Daten würden in einem S3-Bucket gespeichert. Somit könnten die Metadaten schnell abgefragt werden und die 
eigentlichen Daten werden erst On-Demand aus dem entsprechenden Bucket gelesen. Eine prototypische Implementierung ist im Verzeichnis
[mltest/datasets](mltest/datasets) zu finden.

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
        - RandomForest
            - Average Depth
            - Feature Importance
            - Number of Trees
        - Robustness
            - Für NN: [RobustBench](https://github.com/RobustBench/robustbench)
            - Für XGB: https://github.com/laudv/veritas ??
            - Für DT: ???
            - SVM: https://github.com/abstract-machine-learning/saver
        - SVM:
          - TCAT
          - ξ-α Test-
          - Ratio: |#supportvector|/|traing_set|
          - Propensity Score (Joachims)
          - Robustness - https://arxiv.org/pdf/1206.6389.pdf ?
          - Anzahl an Stützvektoren
          - Feature Importance
    - Metriken:
        - Accuracy, Precision, Recall, F1, AUC, ROC, Confusion Matrix, ...
        - MSE, MAE, ...

- Wie wird quantisierung behandelt?
    - Modell wird submitted -> Model -> [Model, Model8Bit]
    - Execute tests for each model..

# "Äußere Schleife"
- Kreuzprodukt Plattformen x Optimierungen (Pruning, Quanitisierung, andere Formate)

Vorgehen:
- Generiere alle "candidatemodels"
- Generiere pipeline für alle modelle ("SELECT")

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
- https://github.com/sbuschjaeger/fastinference
- https://treelite.readthedocs.io/en/latest/