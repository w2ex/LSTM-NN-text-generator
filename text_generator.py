import tensorflow as tf
import os
import sys
import numpy as np
import random
from collections import deque

GAMMA = 0.95
REPLAY_MEMORY = 10000.
BATCH = 32
LOGFILE = "\trumptweets.txt"
FILE  = open(LOGFILE)
data = FILE.read()
FILE.close()
#data = data.lower()
vocabulary = list(set(data))
vocabulary.sort()

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush() #rewrite on the same line
    if iteration == total: 
        print()
        
def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))

    cnt=0
    for s in data_:
        v = [0.0]*len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1

    return [data]
        
def createNetwork(input_size, lstm_size, num_layers, output_size, session):
    
    with tf.name_scope('Input'):
        input = tf.placeholder(tf.float32, shape=(None, None, input_size), name="input")
        lstm_init_value = tf.placeholder(tf.float32, shape=(None, num_layers*2*lstm_size), name="lstm_init_value")
        
    with tf.name_scope('LSTM_layers'):
        lstm_cells = [ tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(num_layers)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=False)
        
        outputs, lstm_new_state = tf.nn.dynamic_rnn(lstm, input, initial_state=lstm_init_value, dtype=tf.float32)
    
    with tf.name_scope('Fully_connected_layer'):
        rnn_out_W = tf.Variable(tf.random_normal( (lstm_size, output_size), stddev=0.01 ))
        rnn_out_B = tf.Variable(tf.random_normal( (output_size, ), stddev=0.01 ))
        
        outputs_reshaped = tf.reshape( outputs, [-1, lstm_size] )
        network_output = ( tf.matmul( outputs_reshaped, rnn_out_W ) + rnn_out_B )
        
        batch_time_shape = tf.shape(outputs)
        final_outputs = tf.reshape( tf.nn.softmax(network_output), (batch_time_shape[0], batch_time_shape[1], output_size))
        
    return lstm_init_value, input, network_output, lstm_new_state, final_outputs
    
def trainNetwork(init_values, input, network_output, lstm_new_state, final_outputs, session):
    
    with tf.name_scope('Target'):
        y_batch = tf.placeholder(tf.float32, (None, None, len(vocabulary)))
        y_batch_long = tf.reshape(y_batch, [-1, len(vocabulary)])
        
    with tf.name_scope('Loss_function'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long))
    
    train_op = tf.train.RMSPropOptimizer(1e-6, 0.9).minimize(cost)
    
    session.run(tf.global_variables_initializer())
    
    
    D = deque()
    
    over = True
    
    L = np.array([[0 for i in range(len(vocabulary))] for j in range (70)])
    last_L = np.copy(L)
    t=0
    
    saver = tf.train.Saver() # for saving the progress
    session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state('text_generation')
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path) # searching for existing network
        print ("Successfully loaded network :", checkpoint.model_checkpoint_path)
    else:
        print ("No network found. Creating a new network initialized randomly...")
    
    print("Learning started...")
    for j in range(1):
        for i, char in enumerate(data) :
                
            last_L = np.copy(L) # updating state variables
            L[0:-1] = L[1:]
            
            L[-1] = [0 for i in range(len(vocabulary))]
            L[-1][vocabulary.index(char)] = 1
    
    
            ## Learning
            if t <= BATCH : # first steps : filling the batch queue
                D.append((last_L, np.copy(L)))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()
                
            else : # batch full enough, we can now start learning
    
                D.append((last_L, np.copy(L))) # continue filling the queue
                
                if len(D) > REPLAY_MEMORY:
                    D.popleft() # pop if the queue is already full
            
                minibatch = random.sample(D, BATCH) # extracting a sample from the batch
    
                last_char_batch = [d[0] for d in minibatch] # extracting each feature from the batch elements
                future_char_batch = [d[1] for d in minibatch]
    
                
                train_op.run(feed_dict = { # applying learning algorithm with the following values
                y_batch: future_char_batch,
                input : last_char_batch,
                init_values : np.zeros((BATCH, 4*128))})
            
            t+=1
            #if t%5000 == 0 :
                #saver.save(session, 'text_generation\save', global_step=t)
            printProgressBar(t, len(data))
    
    saver.save(session, 'text_generation\save', global_step=t)
    
    
    print('Finished learning.')
    print('Generating text.')
    
    seed = "I am "
    seed = seed.lower()
    state = [np.zeros((4*128))]
    

    for i in range(len(seed)) :
        output, state = session.run([final_outputs, lstm_new_state], feed_dict = {input : embed_to_vocab(seed[i], vocabulary), init_values : [state[0]]})
    
    gen_str = "I am "
    for i in range(1000) :
        # if random.random()<0.3 :
        #element=np.argmax(output[0][0])
        # else :
        element = np.random.choice( range(len(vocabulary)), p=(output[0][0]**2)/np.sum(output[0][0]**2))
        gen_str += vocabulary[element]
        output ,state = session.run([final_outputs, lstm_new_state], feed_dict = {input : embed_to_vocab(vocabulary[element], vocabulary), init_values : [state[0]]})
    
    print(gen_str)

def run():
    sess = tf.InteractiveSession()
    print("Session created")
    init_values, input, network_output, lstm_new_state, final_outputs = createNetwork(len(vocabulary), 128, 2, len(vocabulary), sess)
    print("Network ready")
    trainNetwork(init_values, input, network_output, lstm_new_state, final_outputs, sess)
    sess.close()
