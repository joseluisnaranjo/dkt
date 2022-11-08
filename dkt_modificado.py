#
#   Deep Knowledge Tracing (DKT) Implementation
#   Mohammad M H Khajah <mohammad.khajah@colorado.edu>
#   Copyright (c) 2016 all rights reserved.
#
#   How to use:
#       python dkt.py dataset.txt dataset_split.txt
#
#   Script saves 3 files:
#       dataset.txt.model_weights trained model weights
#       dataset.txt.history training history (training LL, test AUC)
#       dataset.txt.preds predictions for test trials
#
import array
import os
import sys
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential #, Graph
from keras.layers.core import  Masking #, LSTM, TimeDistributedDense,
from keras.layers import LSTM, TimeDistributed, Dense
from keras import backend as K
import tensorflow  as tf
from sklearn.metrics import roc_auc_score
import theano.tensor as Th
import random
import math
import argparse
from model import Dktmodel
import reader as rd

overall_loss = [0.0]
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='Dataset to be used', default='assistments.txt', required=True)
    parser.add_argument('--splitfile', type=str, help='Split file', default='assistments_split.txt', required=True)
    parser.add_argument('--hiddenunits', type=int, help='Number of LSTM hidden units.', default=200, required=False)
    parser.add_argument('--batchsize', type=int, help='Number of sequences to process in a batch.', default=32, required=False)
    parser.add_argument('--timesteps', type=int, help='Number of timesteps to process in a batch.', default=100, required=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs.', default=10, required=False)
    args = parser.parse_args()
    
    dataset = args.dataset
    split_file = args.splitfile
    hidden_units = args.hiddenunits
    batch_size = args.batchsize
    timesteps = args.timesteps
    epochs = args.epochs
    
    model_file = dataset + '.model_weights'
    history_file = dataset + '.history'
    preds_file = dataset + '.preds'
    
    overall_loss = [0.0]
    preds = []
    history = []
    
    # load dataset
    training_seqs, testing_seqs, num_skills = rd.load_dataset(dataset, split_file)
    print ("Training Sequences: %d" % len(training_seqs))
    print ("Testing Sequences: %d" % len(testing_seqs))
    print ("Number of skills: %d" % num_skills)


    ##############**


    max_input_sequence= max(len(seq) for seq in training_seqs)
    max_output_sequence= max(len(seq) for seq in testing_seqs)

    print('max_input_sequence: ', max_input_sequence)
    print('max_output_sequence: ', max_output_sequence)

    
    dktmodel = Dktmodel(batch_size,timesteps,num_skills, hidden_units)
    model = dktmodel.Iniciar()
    # training function
    def trainer(X, Y):        
        overall_loss[0] += model.train_on_batch(X, Y)[0]
    
    # prediction
    def predictor(X, Y):
        batch_activations = model.predict_on_batch(X)
        skill = Y[:,:,0:num_skills]
        obs = Y[:,:,num_skills]
        y_pred = np.squeeze(np.array(batch_activations))
        
        rel_pred = np.sum(y_pred * skill, axis=2)
        
        for b in range(0, X.shape[0]):
            for t in range(0, X.shape[1]):
                if X[b, t, 0] == -1.0:
                    continue
                preds.append((rel_pred[b][t], obs[b][t]))
        
    # call when prediction batch is finished
    # resets LSTM state because we are done with all sequences in the batch
    def finished_prediction_batch(percent_done):
        model.reset_states()
        
    # similiar to the above
    def finished_batch(percent_done):
        print ("(%4.3f %%) %f" % (percent_done, overall_loss[0]))
        model.reset_states()
        
    # run the model
    for e in range(0, epochs):
        model.reset_states()
        
        # train
        run_func(training_seqs, num_skills, trainer, batch_size, timesteps, finished_batch, model=model)
        
        model.reset_states()
        
        # test
        run_func(testing_seqs, num_skills, predictor, batch_size, timesteps, finished_prediction_batch, model=model)
        
        # compute AUC
        auc = roc_auc_score([p[1] for p in preds], [p[0] for p in preds])
        
        # log
        history.append((overall_loss[0], auc))
        
        # save model
        model.save_weights(model_file, overwrite=True)
        print ("==== Epoch: %d, Test AUC: %f" % (e, auc))
        
        # reset loss
        overall_loss[0] = 0.0
        
        # save predictions
        with open(preds_file, 'w') as f:
            f.write('was_heldout\tprob_recall\tstudent_recalled\n')
            for pred in preds:
                f.write('1\t%f\t%d\n' % (pred[0], pred[1]))
        
        with open(history_file, 'w') as f:
            for h in history:
                f.write('\t'.join([str(he) for he in h]))
                f.write('\n')
                
        # clear preds
        preds = []


def trainer(X, Y, model):
        #dktmodel.loss_function(X,Y)
        overall_loss[0] += model.train_on_batch(X, Y)[0]

def run_func(seqs, num_skills, funsion, batch_size, timesteps, batch_done = None, model = None):

    assert(min([len(s) for s in seqs]) > 0)
    
    # randomize samples
    seqs = seqs[:]
    #random.shuffle(seqs)
    
    processed = 0
    for start_from in range(0, len(seqs), batch_size): # range  have 3 args, 1arg tells from where to start, 2nd arg: until where to go, 3rd arg: in steps of size
       end_before = min(len(seqs), start_from + batch_size) #min function returns the min val if it get 1 arg or the min arg if it gets more than 1 args
       x = []
       y = []
       # until here we have the first slice of 32 secunaces each one of different lengh 
       for seq in seqs[start_from:end_before]: #why seq starts whith other value in this case 2
           x_seq = []
           y_seq = []
           xt_zeros = [0 for i in range(0, num_skills*2)] # un vector de ceros del doble de la longitud del numero de skills, en este caso 248
           ct_zeros = [0 for i in range(0, num_skills+1)] # vector de ceros del numeor de skills mas una 125 , nose xq???, correspondera a el numero de ejecicio y alvalor final indica si es correcto o no
           xt = xt_zeros[:]
           for skill, is_correct in seq:
               
               
               ct = ct_zeros[:]
               ct[skill] = 1
               ct[num_skills] = is_correct
               y_seq.append(ct)
               
               # one hot encoding of (last_skill, is_correct)
               pos = skill * 2 + is_correct
               xt = xt_zeros[:]
               xt[pos] = 1

               x_seq.append(xt)  # aqui agrego cada interaccion (como one hot encoding, indicando la posision de la habilidad por dos ya que la habilidad 1 esta indicando en las dos primeras posisiones como verdadreo o falso)
           x.append(x_seq) # el x_ sex es in vectotor de 32 elementos (batchsize) cada elemnto es un vector de el doble de skills indicando T or F
           y.append(y_seq) # el y_seq es un vecors de 32 elemntos cada uno de 124 elemntos(skill + 1), el calor en la posisino n skill es uno y el ultimo valors es 1/0 dependiendo de si esta habilidad fue respondoida correctament
       
       maxlen = max([len(s) for s in x])
       maxlen = round_to_multiple(maxlen, timesteps)
       # fill up the batch if necessary
       if len(x) < batch_size: # x es una lista de 32(batch_size) sequencias, cada secuancia es una lista de interacciones, y cada interaccion es un array de 248 values( 1 hot encod)
            for e in range(0, batch_size - len(x)):
                x_seq = []
                y_seq = []
                for t in range(0, timesteps):
                    x_seq.append([-1.0 for i in range(0, num_skills*2)])
                    y_seq.append([0.0 for i in range(0, num_skills+1)])
                x.append(x_seq)
                y.append(y_seq)
        
       X = pad_sequences(x, padding='post', maxlen = maxlen, dim=num_skills*2, value=-1.0)
       Y = pad_sequences(y, padding='post', maxlen = maxlen, dim=num_skills+1, value=-1.0)


       # A QUI ES DONDE SE EMPIESA
       #    funsion, representa training the model o evaluationg the model

        
       for t in range(0, maxlen, timesteps):
           trainer(X[:,t:(t+timesteps),:], Y[:,t:(t+timesteps),:], model) 
           
       processed += end_before - start_from
       
       # reset the states for the next batch of sequences
       if batch_done:
           batch_done((processed * 100.0) / len(seqs))
   
def round_to_multiple(x, base):
    return int(base * math.ceil(float(x)/base))

def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32',
    padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

if __name__ == "__main__":
    main()
    