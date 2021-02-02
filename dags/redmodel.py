# Code that describes the Airflow pipeline:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import gzip as gzip_lib
import numpy as np
import os
import pprint
import shutil
import tempfile
import urllib
import uuid
import datetime
import glob

import pandas as pd

pp = pprint.PrettyPrinter()

import tensorflow as tf
import neural_structured_learning as nsl

import tfx
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Examples

from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

from typing import Text

import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
import tensorflow_hub as hub



_pipeline_name = 'redmodel'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_mh_root = os.path.join(os.environ['HOME'], 'redmodel')
_airflow_root = os.path.join(os.environ['HOME'], 'airflow')

_data_root = os.path.join(_mh_root, 'data', 'simple')
# Python module file to inject customized logic into the TFX components. The
_transform_module_file = os.path.join(_airflow_root,'utils', 'transform.py')
_trainer_module_file = os.path.join(_airflow_root,'utils', 'trainer.py')
_module_file = os.path.join(_airflow_root,'utils', 'module.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_mh_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_mh_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_mh_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2020, 1, 1),
}

def _load_csv_dataset(path):
    csv_data = pd.read_csv(path)
    return csv_data.values

train_set = _load_csv_dataset(_data_root+"/train/train.csv")
unsup_set = _load_csv_dataset(_data_root+"/unsup/unsup.csv")
eval_set = _load_csv_dataset(_data_root+"/eval/test.csv")

examples_path = tempfile.mkdtemp(prefix="tfx-data")
train_path = os.path.join(examples_path, "train.tfrecord")
eval_path = os.path.join(examples_path, "eval.tfrecord")
unsup_path = os.path.join(examples_path, "unsup.tfrecord")


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(label, text):
      # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type for supervised examples.
    feature = {
      'label': _int64_feature(label),
      'text': _bytes_feature(text),
      }
  # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_example_unsup(text):
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type for supervised examples.
    feature = {
      'text': _bytes_feature(text),
      }
  # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


for path, dataset in [(train_path, train_set), (eval_path, eval_set)]:
    with tf.io.TFRecordWriter(path) as writer:
        for example in dataset:
            features, label = example[:-1], example[-1]
            writer.write(
                serialize_example(
                    label=label, text=features
                ))

for path, dataset in [(unsup_path, unsup_set)]:
    with tf.io.TFRecordWriter(path) as writer:
        for example in dataset:
            feature = example[1]
            writer.write(
                serialize_example_unsup(
                    text=features
                ))

def _write_train_tfrecord():
    filename = [train_path, unsup_path]
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(train_path)


def make_example_with_unique_id(example, id_feature_name):
  """Adds a unique ID to the given `tf.train.Example` proto.

  This function uses Python's 'uuid' module to generate a universally unique
  identifier for each example.

  Args:
    example: An instance of a `tf.train.Example` proto.
    id_feature_name: The name of the feature in the resulting `tf.train.Example`
      that will contain the unique identifier.

  Returns:
    A new `tf.train.Example` proto that includes a unique identifier as an
    additional feature.
  """
  result = tf.train.Example()
  result.CopyFrom(example)
  unique_id = uuid.uuid4()
  result.features.feature.get_or_create(
      id_feature_name).bytes_list.MergeFrom(
          tf.train.BytesList(value=[str(unique_id).encode('utf-8')]))
  return result


@component
def IdentifyExamples(orig_examples: InputArtifact[Examples],
                     identified_examples: OutputArtifact[Examples],
                     id_feature_name: Parameter[str],
                     component_name: Parameter[str]) -> None:

    # Get a list of the splits in input_data
    splits_list = artifact_utils.decode_split_names(
        split_names=orig_examples.split_names)

    for split in splits_list:
        input_dir = os.path.join(orig_examples.uri, split)
        output_dir = os.path.join(identified_examples.uri, split)
        os.mkdir(output_dir)
        with beam.Pipeline() as pipeline:
            (pipeline
            | 'ReadExamples' >> beam.io.ReadFromTFRecord(
                os.path.join(input_dir, '*'),
                coder=beam.coders.coders.ProtoCoder(tf.train.Example))
            | 'AddUniqueId' >> beam.Map(make_example_with_unique_id, id_feature_name)
            | 'WriteIdentifiedExamples' >> beam.io.WriteToTFRecord(
                file_path_prefix=os.path.join(output_dir, 'data_tfrecord'),
                coder=beam.coders.coders.ProtoCoder(tf.train.Example),
                file_name_suffix='.gz'))
    identified_examples.split_names = artifact_utils.encode_split_names(
        splits=splits_list)

    return
  
swivel_url = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(swivel_url, input_shape=[], dtype=tf.string)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_embedding_example(example):
    """Create tf.Example containing the sample's embedding and its ID."""
    sentence_embedding = hub_layer(tf.sparse.to_dense(example['text']))

    # Flatten the sentence embedding back to 1-D.
    sentence_embedding = tf.reshape(sentence_embedding, shape=[-1])

    feature_dict = {
        'id': _bytes_feature(tf.sparse.to_dense(example['id']).numpy()),
        'embedding': _float_feature(sentence_embedding.numpy().tolist())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def create_dataset(uri):
    tfrecord_filenames = [os.path.join(uri, name) for name in os.listdir(uri)]
    return tf.data.TFRecordDataset(tfrecord_filenames, compression_type='GZIP')


def create_embeddings(train_path, output_path):
    dataset = create_dataset(train_path)
    embeddings_path = os.path.join(output_path, 'embeddings.tfr')

    feature_map = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.VarLenFeature(tf.string),
        'text': tf.io.VarLenFeature(tf.string)
    }

    with tf.io.TFRecordWriter(embeddings_path) as writer:
        for tfrecord in dataset:
            tensor_dict = tf.io.parse_single_example(tfrecord, feature_map)
            embedding_example = create_embedding_example(tensor_dict)
            writer.write(embedding_example.SerializeToString())


def build_graph(output_path, similarity_threshold):
    embeddings_path = os.path.join(output_path, 'embeddings.tfr')
    graph_path = os.path.join(output_path, 'graph.tfv')
    nsl.tools.build_graph([embeddings_path], graph_path, similarity_threshold)


"""Custom Artifact type"""


class SynthesizedGraph(tfx.types.artifact.Artifact):
    """Output artifact of the SynthesizeGraph component"""
    TYPE_NAME = 'SynthesizedGraphPath'
    PROPERTIES = {
        'span': standard_artifacts.SPAN_PROPERTY,
        'split_names': standard_artifacts.SPLIT_NAMES_PROPERTY,
    }

@component
def SynthesizeGraph(identified_examples: InputArtifact[Examples],
                    synthesized_graph: OutputArtifact[SynthesizedGraph],
                    similarity_threshold: Parameter[float],
                    component_name: Parameter[str]) -> None:

    # Get a list of the splits in input_data
    splits_list = artifact_utils.decode_split_names(
        split_names=identified_examples.split_names)

    # We build a graph only based on the 'train' split which includes both
    # labeled and unlabeled examples.
    train_input_examples_uri = os.path.join(identified_examples.uri, 'train')
    print(train_input_examples_uri, 'These')
    output_graph_uri = os.path.join(synthesized_graph.uri, 'train')
    os.mkdir(output_graph_uri)

    print('Creating embeddings...')
    create_embeddings(train_input_examples_uri, output_graph_uri)

    print('Synthesizing graph...')
    build_graph(output_graph_uri, similarity_threshold)

    synthesized_graph.split_names = artifact_utils.encode_split_names(
        splits=['train'])

    return
  
def split_train_and_unsup(input_uri):
    'Separate the labeled and unlabeled instances.'

    tmp_dir = tempfile.mkdtemp(prefix='tfx-data')
    tfrecord_filenames = [
        os.path.join(input_uri, filename) for filename in os.listdir(input_uri)
    ]
    train_path = os.path.join(tmp_dir, 'train.tfrecord')
    unsup_path = os.path.join(tmp_dir, 'unsup.tfrecord')
    with tf.io.TFRecordWriter(train_path) as train_writer, \
        tf.io.TFRecordWriter(unsup_path) as unsup_writer:
        for tfrecord in tf.data.TFRecordDataset(
            tfrecord_filenames, compression_type='GZIP'):
                example = tf.train.Example()
                example.ParseFromString(tfrecord.numpy())
        if ('label_xf' not in example.features.feature or
            example.features.feature['label_xf'].int64_list.value[0] == -1):
            writer = unsup_writer
        else:
            writer = train_writer
        writer.write(tfrecord.numpy())
    return train_path, unsup_path

def gzip(filepath):
    with open(filepath, 'rb') as f_in:
        with gzip_lib.open(filepath + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filepath)


def copy_tfrecords(input_uri, output_uri):
    for filename in os.listdir(input_uri):
        input_filename = os.path.join(input_uri, filename)
        output_filename = os.path.join(output_uri, filename)
        shutil.copyfile(input_filename, output_filename)


@component
def GraphAugmentation(identified_examples: InputArtifact[Examples],
                      synthesized_graph: InputArtifact[SynthesizedGraph],
                        augmented_examples: OutputArtifact[Examples],
                        num_neighbors: Parameter[int],
                        component_name: Parameter[str]) -> None:

    # Get a list of the splits in input_data
    splits_list = artifact_utils.decode_split_names(
        split_names=identified_examples.split_names)

    train_input_uri = os.path.join(identified_examples.uri, 'train')
    eval_input_uri = os.path.join(identified_examples.uri, 'eval')
    train_graph_uri = os.path.join(synthesized_graph.uri, 'train')
    train_output_uri = os.path.join(augmented_examples.uri, 'train')
    eval_output_uri = os.path.join(augmented_examples.uri, 'eval')

    os.mkdir(train_output_uri)
    os.mkdir(eval_output_uri)

    # Separate out the labeled and unlabeled examples from the 'train' split.
    train_path, unsup_path = split_train_and_unsup(train_input_uri)


    output_path = os.path.join(train_output_uri, 'nsl_train_data.tfr')
    pack_nbrs_args = dict(
        labeled_examples_path=train_path,
        unlabeled_examples_path=unsup_path,
        graph_path=os.path.join(train_graph_uri, 'graph.tfv'),
        output_training_data_path=output_path,
        add_undirected_edges=True,
        max_nbrs=num_neighbors)
    print('nsl.tools.pack_nbrs arguments:', pack_nbrs_args)
    nsl.tools.pack_nbrs(**pack_nbrs_args)

    # Downstream components expect gzip'ed TFRecords.
    gzip(output_path)

    # The test examples are left untouched and are simply copied over.
    copy_tfrecords(eval_input_uri, eval_output_uri)

    augmented_examples.split_names = identified_examples.split_names

    return
    
def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     metadata_path: Text,
                     direct_num_workers: int) -> pipeline.Pipeline:


    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train.tfrecord'),
        example_gen_pb2.Input.Split(name='eval', pattern='eval.tfrecord')
    ])
    
    example_gen = ImportExampleGen(input_base=examples_path, input_config=input_config)

    identify_examples = IdentifyExamples(
        orig_examples=example_gen.outputs['examples'],
        component_name=u'IdentifyExamples',
        id_feature_name=u'id')

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(
        examples=identify_examples.outputs["identified_examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    synthesize_graph = SynthesizeGraph(
        identified_examples=identify_examples.outputs['identified_examples'],
        component_name=u'SynthesizeGraph',
        similarity_threshold=0.99)

    transform = Transform(
        examples=identify_examples.outputs['identified_examples'],
        schema=schema_gen.outputs['schema'],
        # TODO(b/169218106): Remove transformed_examples kwargs after bugfix is released.
        transformed_examples=channel.Channel(
            type=standard_artifacts.Examples,
            artifacts=[standard_artifacts.Examples()]),
        module_file=_transform_module_file)

    # Augments training data with graph neighbors.
    graph_augmentation = GraphAugmentation(
        identified_examples=transform.outputs['transformed_examples'],
        synthesized_graph=synthesize_graph.outputs['synthesized_graph'],
        component_name=u'GraphAugmentation',
        num_neighbors=3)

    trainer = Trainer(
        module_file=_trainer_module_file,
        transformed_examples=graph_augmentation.outputs['augmented_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, identify_examples, statistics_gen, schema_gen, validate_stats, synthesize_graph, transform, graph_augmentation, trainer
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers])


# 'DAG' below need to be kept for Airflow to detect dag.
DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
        # 0 means auto-detect based on on the number of CPUs available during
        # execution time.
        direct_num_workers=0))
