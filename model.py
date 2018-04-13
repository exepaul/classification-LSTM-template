import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

words=np.load('words_list_lstm.npy')
word_embedding=np.load('word_embedding_lstm.npy')

class LSTMclassifier():

    def __init__(self,hdim,lables):
        
        tf.reset_default_graph()

        #placeholders
        input_x = tf.placeholder(tf.int32,shape=[None,None])
        output_y = tf.placeholder(tf.int32,shape=[None,])


        self.placeholder={'input':input_x,'output':output_y}

        #word_embedding
        word_embedd = tf.get_variable('embedding',shape=[400000,100],dtype=tf.float32,initializer=tf.constant_initializer(np.array(word_embedding)),trainable=False)
        embedding_lookup = tf.nn.embedding_lookup(word_embedd,input_x)

        #sequence_length
        sequence_le = tf.count_nonzero(input_x,axis=-1)

        #model
        with tf.variable_scope('encoder') as scope:
            cell=rnn.LSTMCell(num_units=hdim)
            model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=embedding_lookup,sequence_length=sequence_le,dtype=tf.float32)


        final_output,(fs,fc)=model

        #transform_output
        final_output_forward = tf.transpose(final_output[0],[1,0,2])
        final_output_backward = tf.transpose(final_output[1],[1,0,2])

        state_output = tf.concat([fs.c,fc.c],axis=-1)
        final_output_both = tf.concat([final_output_forward[0],final_output_backward[0]],axis=-1)


        #weights and fully_connected_layer
        weights=tf.get_variable('weights',shape=[2*hdim,lables],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        bias = tf.get_variable('bias',shape=[lables],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        #logits ( final_output_matrix )

        logits_ = tf.matmul(final_output_both,weights) + bias

        #normalization
        prob = tf.nn.softmax(logits_)
        pred=tf.argmax(prob,axis=-1)

        #cross_entropy
        ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_,labels=output_y)
        loss= tf.reduce_mean(ce)

        #accuracy
        accuracy = tf.reduce_mean(
                   tf.cast(
                   (tf.equal(
                    tf.cast(pred,tf.int32),output_y)),
                    tf.float32))

        #training
        training_ = tf.train.AdamOptimizer().minimize(loss)




        self.out ={'model':model,
                   'output':final_output_both,
                   'logits':logits_,
                   'prob':prob,
                   'pred':pred,
                   'ce':ce,
                   'loss':loss,
                   'accuracy':accuracy,
                   'train':training_
                   }




        self.shape_checking = {'embedding_l':embedding_lookup,   #2x4x100
                               'final_output':final_output,      #2x4x10
                               'fs':fs,                          #2x10
                               'transpo':final_output_backward,  #4x2x10
                               'for':final_output_forward,       #4x2x10
                               'state':state_output,             #2x20
                               'fina':final_output_both,         #2x20
                               'weig':weights,                   #20x10
                               'log':logits_}                    #2x10



#checking model with dummy data


def rand_exe(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(model.shape_checking,feed_dict={model.placeholder['input']:np.random.randint(0,9,[2,4]),model.placeholder['output']:np.random.randint(0,9,[2])})



if __name__=='__main__':
    model=LSTMclassifier(10,10)
    out=rand_exe(model)
    print(out['embedding_l'].shape,out['final_output'][0].shape,out['fs'].c.shape,out['transpo'].shape,out['for'].shape,out['state'].shape,out['fina'].shape,out['weig'].shape,out['log'].shape)




#output
# (2, 4, 100) (2, 4, 10) (2, 10) (4, 2, 10) (4, 2, 10) (2, 20) (2, 20) (20, 10) (2, 10)
