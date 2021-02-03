# Redmodel
TFX implementation of text sentiment classification using Airflow.

![Screenshot semisupervised learning workflow](https://raw.githubusercontent.com/devlp121/redmodel-pipeline/master/test1.png)

Pipeline components are implemented as DAGs using Airflow

## ImportExampleGen
This component ingests CSV data into the machine learning pipeline by converting the data types into compatible datatypes.

## IdentifyExample
Marking each example in the utilized data occurs at this DAG component by assigning each instance with a unique identifier

## StatisticsGen


