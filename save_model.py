#! /usr/bin/env python

import tensorflow as tf
import os
import shutil

# Parameters
# ==================================================
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Saving Parameters
tf.flags.DEFINE_integer("model_version", 1, "model version")
tf.flags.DEFINE_string("pb_save_path", "./gec_classification_model", "pb save file.")
tf.flags.DEFINE_string("checkpoint_path", "./runs", "ckpt path")

FLAGS = tf.flags.FLAGS


def export():
    checkpoint_file = FLAGS.checkpoint_path
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_tags = graph.get_operation_by_name("input_tags").outputs[0]
            input_name_entity = graph.get_operation_by_name("input_name_entities").outputs[0]

            #########################
            input_dropout_keep_1 = graph.get_operation_by_name("dropout_keep_prob_1").outputs[0]
            input_dropout_keep_2 = graph.get_operation_by_name("dropout_keep_prob_2").outputs[0]
            input_tempreture = graph.get_operation_by_name("Tempreture").outputs[0]
            input_training = graph.get_operation_by_name("is_training").outputs[0]
            ##########################

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            prediction = graph.get_operation_by_name("output/predictions").outputs[0]

            model_version = FLAGS.model_version
            version_export_path = os.path.join(FLAGS.pb_save_path, str(model_version))
            if os.path.exists(version_export_path):
                shutil.rmtree(version_export_path)
            print("Exporting trained model to ", version_export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(version_export_path)

            tensor_input_data_words = tf.saved_model.utils.build_tensor_info(input_x)
            tensor_input_data_tags = tf.saved_model.utils.build_tensor_info(input_tags)
            tensor_input_data_name_entity = tf.saved_model.utils.build_tensor_info(input_name_entity)

            ###########################
            tensor_input_dropout_keep_1 = tf.saved_model.utils.build_tensor_info(input_dropout_keep_1)
            tensor_input_dropout_keep_2 = tf.saved_model.utils.build_tensor_info(input_dropout_keep_2)
            tensor_input_tempreture = tf.saved_model.utils.build_tensor_info(input_tempreture)
            tensor_input_training = tf.saved_model.utils.build_tensor_info(input_training)
            ###########################

            tensor_output_scores = tf.saved_model.utils.build_tensor_info(scores)
            tensor_output_prediction = tf.saved_model.utils.build_tensor_info(prediction)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_data_words': tensor_input_data_words,
                            "input_data_tags": tensor_input_data_tags,
                            "input_data_name_entity": tensor_input_data_name_entity,
                            "input_dropout_keep_1": tensor_input_dropout_keep_1,
                            "input_dropout_keep_2": tensor_input_dropout_keep_2,
                            "input_tempreture": tensor_input_tempreture,
                            "input_training": tensor_input_training},
                    outputs={'output_scores': tensor_output_scores,
                             'output_prediction': tensor_output_prediction},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'gec_classify': prediction_signature
                }
            )
            builder.save()
            print('Done exporting!')


if __name__ == "__main__":
    export()

