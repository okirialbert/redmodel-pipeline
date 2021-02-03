# Redmodel
TFX implementation of text sentiment classification using Airflow.

![Screenshot semisupervised learning workflow](https://raw.githubusercontent.com/devlp121/redmodel-pipeline/master/test1.png)

Pipeline components are implemented as DAGs using Airflow

#### ImportExampleGen
This component ingests CSV data into the machine learning pipeline by converting the data types into compatible datatypes.

#### IdentifyExample
Marking each example in the utilized data occurs at this DAG component by assigning each instance with a unique identifier

#### StatisticsGen
This component computes statistical information concerning the ingested dataset and the information is used in following stages to evaluate the data for anomalies and also analyze model performance.

#### SchemaGen
This component generates the schema from the provided data and the data is stored in Metadata store for future analysis and evaluation of model performance.

#### ExampleValidator
Anomalies are detected at this stage to determine any inconsistencies in the data that might undermine the performance of the generated machine learning model.

#### SynthesizeGraph 
Neural Structured Learning is utilized in this component to generate a graph using similarity measurement between labelled and unlabelled data.
Embedding for the text is also generated from transfer learning libraries using pretrained layers.

#### Transform
This component performs the general preprocessing tasks that enhance performance and structure the data better.

#### GraphAugmentation
This component retrieves the probable label assignment from the synthesized graph to create a matching example entity that contains both labels and features that will be used during training.

#### Trainer
This method leverages tensorflow to perform training task that completes the model development phase.


## Serving
The Tensorflow Serving library has REST API endpoint features for SavedModels that are used to perform inferences

### Environment Setup
The 
