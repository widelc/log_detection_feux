"""
#################################
 Classification after training the Model, modules and methods in this file evaluate the performance of the trained
 model over the test dataset
 Test Data: Item (8) on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs 
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
################################
"""
#########################################################
# import libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from plotdata import plot_confusion_matrix
from config import Config_classification
from config import new_size

batch_size = Config_classification.get('batch_size')
image_size = (new_size.get('width'), new_size.get('height'))
epochs = Config_classification.get('Epochs')


#########################################################
# Function definition

def classify():
    """
    This function load the trained model from the previous task and evaluates the performance of that over the test
    data set.
    :return: None, Plot the Confusion matrix for the test data on the binary classification
    """
    
    
    ###########################################################################
#    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#        "frames/Test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True
#    )
#    for i in range(1,46):
#        print('Epoch %d' % i)
#        model_file = 'C:/Users/wdeli/Desktop/Test AI/Modele_epochs/32_batch_size/save_at_%d.h5' % i
#        best_model_fire = load_model(model_file)
#        results_eval = best_model_fire.evaluate(test_ds, batch_size=batch_size)
#        print('')
#        
#    #Essayer de faire avec un batch_size = 1 après !
    model_fire = load_model('C:/Users/wdeli/Desktop/Test AI/Modele_epochs/32_batch_size/save_at_42.h5')
    #model_fire = load_model('C:/Users/wdeli/Desktop/Test AI/Output/Models/model_fire_resnet_weighted_40_no_metric_simple/saved_model_32_bs')
    fire_len = int(len(tf.io.gfile.listdir("frames/Test/Fire"))/2)
    no_fire_len = int(len(tf.io.gfile.listdir("frames/Test/No_Fire"))/2)
    ff_array, fnf_array = [], []
    rf, ff, rnf, fnf = 0, 0, 0, 0
    for i in range(0,fire_len):
        img = keras.preprocessing.image.load_img(
                "frames/Test/Fire/resized_test_fire_frame%d.jpg" % i, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model_fire.predict(img_array)
        score = predictions[0]
        if 1-score >= 0.6:
            rf += 1
        else:
            ff += 1
            ff_array.append(i)
        print(i)
            
    for i in range(0,no_fire_len):
        img = keras.preprocessing.image.load_img(
                "frames/Test/No_Fire/resized_test_nofire_frame%d.jpg" % i, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model_fire.predict(img_array)
        score = predictions[0]
        if 1-score <= 0.6:
            rnf += 1
        else:
            fnf += 1
            fnf_array.append(i)
        print(i)
    
    cm = np.array([[rf, ff], [fnf, rnf]], dtype=int)
    cm_plot_labels = ['Fire', 'No Fire']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    
    fire_perf    = 100*rf/(rf+ff)
    no_fire_perf = 100*rnf/(rnf+fnf)
    overall_perf = 100*(rf+rnf)/(rf+rnf+ff+fnf)
    print('------------ Model Performance -----------')
    print('Fire performance :    %.2f percent' %fire_perf)
    print('No-Fire performance : %.2f percent' %no_fire_perf)
    print('Overall performance : %.2f percent' %overall_perf)
    
    return ff_array, fnf_array
            
    
    ###########################################################################
    
    
    
#    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#        "frames/Test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True
#    )
#
#    model_fire = load_model('C:/Users/wdeli/Desktop/Test AI/Output/Models/model_fire_resnet_weighted_40_no_metric_simple/saved_model_32_bs')
#
#    _ = model_fire.evaluate(test_ds, batch_size=batch_size)
#
#    best_model_fire = load_model('C:/Users/wdeli/Desktop/Test AI/Modele_epochs/32_batch_size/save_at_42.h5')
#    results_eval = best_model_fire.evaluate(test_ds, batch_size=batch_size)
#
#    for name, value in zip(model_fire.metrics_names, results_eval):
#        print(name, ': ', value)
#    print()
#
#    #cm = np.array([[results_eval[1], results_eval[4]], [results_eval[2], results_eval[3]]])
#    cm_plot_labels = ['Fire', 'No Fire']
#    #plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
#
#    model_file = 'C:/Users/wdeli/Desktop/Test AI/Modele_epochs/32_batch_size/save_at_%d.h5' % 42
#    model_fire = load_model(model_file)
#    test_fire_ds = tf.keras.preprocessing.image_dataset_from_directory(
#        "frames/confusion_test/Fire_test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True)
#    test_no_fire_ds = tf.keras.preprocessing.image_dataset_from_directory(
#        "frames/confusion_test/No_Fire_test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True)
#    fire_eval = model_fire.evaluate(test_fire_ds)
#    no_fire_eval = model_fire.evaluate(test_no_fire_ds)
#    true_fire = len(tf.io.gfile.listdir("frames/confusion_test/Fire_test/Fire"))/2
#    true_no_fire = len(tf.io.gfile.listdir("frames/confusion_test/No_Fire_test/No_Fire"))/2
#    tp = fire_eval[1] * true_fire
#    fp = (1 - fire_eval[1]) * true_fire
#    tn = (1 - no_fire_eval[1]) * true_no_fire
#    fn = no_fire_eval[1] * true_no_fire
#    cm = np.array([[tp, fn], [fp, tn]], dtype=int)
#    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
