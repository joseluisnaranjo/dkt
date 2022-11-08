from keras.models import Sequential #, Graph
from keras.layers.core import  Masking #, LSTM, TimeDistributedDense,
from keras.layers import LSTM, TimeDistributed, Dense
import theano.tensor as Th
from keras import backend as K
import tensorflow as tf

class Dktmodel:
    

    def __init__(self, batch_size, timesteps, num_skills, hidden_units):    
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.num_skills = num_skills
        self.hidden_units = hidden_units
        self.batch_input_shape = (batch_size, timesteps, num_skills*2)
    
    # Our loss function
    # The model gives predictions for all skills so we need to get the 
    # prediction for the skill at time t. We do that by taking the column-wise
    # dot product between the predictions at each time slice and a
    # one-hot encoding of the skill at time t.
    # y_true: (nsamples x nsteps x nskills+1) 
    # y_pred: (nsamples x nsteps x nskills) 
    def loss_function(self, y_true, y_pred):
        #skill = K.cast_to_floatx(y_true[:,:,0:num_skills])
        #obs = K.cast_to_floatx(y_true[:,:,num_skills])
        skill = y_true[:,:,0:self.num_skills]
        obs = y_true[:,:,self.num_skills]

        print ( f'y_true: {y_true}')
        print (y_true.shape) # y_true: Tensor("loss_function/strided_slice:0", shape=(32, 100, 124), dtype=int32) (32, 100, 125)
        #print (y_true.type)
        #tf.dtypes.cast(y_pred, tf.int32)
        print ( f'y_pred: {y_pred}')
        #print (tf.debugging.assert_type(y_pred, tf_type= tf.int32)) # y_pred: Tensor("loss_function/strided_slice:0", shape=(32, 100, 124), dtype=int32) (32, 100, 124)
        #print (y_pred.type)
        otro = tf.dtypes.cast(skill, tf.float32)
        #print ( f'otro: {otro}')
        print ( f'skill: {skill}')
        #print (tf.debugging.assert_type(skill, tf_type= tf.int32)) # skill: Tensor("loss_function/strided_slice:0", shape=(32, 100, 124), dtype=int32)  (32, 100, 124)
        #print (skill.type)
        print ( f'obs: {obs}')
        print (obs.shape) # obs: Tensor("loss_function/strided_slice:0", shape=(32, 100, 124), dtype=int32) (32, 100)
        #print (obs.type)
        

        multiplica = tf.math.multiply(y_pred , otro)

        print ( f'multiplica: {multiplica}')


       
        print (multiplica.numpy())

        print ("************************************************************************") 
        

        rel_pred = Th.sum(multiplica, axis=2)


        print ("************************************************************************") 
        
        # keras implementation does a mean on the last dimension (axis=-1) which
        # it assumes is a singleton dimension. But in our context that would
        # be wrong.
        return K.binary_crossentropy(rel_pred, obs)

    def Iniciar(self):
    # build model
        model = Sequential()
        a = self.batch_input_shape

        # ignore padding
        model.add(Masking(-1.0, batch_input_shape = a ))
        # not included originally;     n_hidden = 32  # size of hidden layer in LSTM
        # lstm configured to keep states between batches
        model.add(LSTM( self.hidden_units, input_dim = self.num_skills*2, 
                        #output_dim = #hidden_units, 
                        return_sequences=True,
                        batch_input_shape = self.batch_input_shape,
                        stateful = True
        ))


        # readout layer. TimeDistributedDense uses the same weights for all
        # time steps.
        model.add(TimeDistributed(Dense( self.num_skills))) #,input_dim = hidden_units, was taken out this parameter

        # optimize with rmsprop which dynamically adapts the learning
        # rate of each weight.
        model.compile(loss=self.loss_function,optimizer='rmsprop', run_eagerly=True) #,class_mode="binary" arg taken otu
        return model

