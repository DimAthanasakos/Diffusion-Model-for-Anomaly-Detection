import numpy as np
import yaml
import os
from Models.Diffusion import GSGM
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import utils


torch.random.set_seed(1233)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='config.yaml', help='Config file with training parameters')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--data_path', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Path containing the training files')
    parser.add_argument('--load', action='store_true', default=False,help='Load trained model')
    parser.add_argument('--large', action='store_true', default=False,help='Train with a large model')

    flags = parser.parse_args()
    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)
      
    
    data_size, training_data, test_data = utils.DataLoader(flags.data_path,
                                                           flags.file_name,
                                                           flags.npart,
                                                           n_events = config['n_events'],
                                                           rank = hvd.rank(), 
                                                           size= hvd.size(),
                                                           batch_size = config['BATCH'],
                                                           norm=config['NORM'])

    model = GSGM(config=config,npart=flags.npart)
 
    model_name = config['MODEL_NAME']
    if flags.large:
        model_name+='_large'
    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
        
        
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=config['LR']*hvd.size(),
        decay_steps=config['MAXEPOCH']*int(data_size/config['BATCH'])
    )

    #opt = tf.keras.optimizers.Adam(learning_rate=config['LR']*hvd.size())
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)

        
    model.compile(            
        optimizer=opt,
        #run_eagerly=True,
        experimental_run_tf_function=False,
        weighted_metrics=[])

    if flags.load:
        model.load_weights(checkpoint_folder).expect_partial()

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=50,restore_best_weights=True),
    ]

        
    if hvd.rank()==0:
        checkpoint = ModelCheckpoint(checkpoint_folder,mode='auto',
                                     save_best_only=True,
                                     save_weights_only=True, 
                                     save_freq='epoch',)  # Save after each epoch
        callbacks.append(checkpoint)
        
    
    history = model.fit(
        training_data,
        epochs=config['MAXEPOCH'],
        callbacks=callbacks,
        steps_per_epoch=int(data_size/config['BATCH']),
        validation_data=test_data,
        validation_steps=int(data_size*0.1/config['BATCH']),
        verbose=2 if hvd.rank()==0 else 0,
        #steps_per_epoch=1,
    )

    
