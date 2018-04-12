import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn



words=np.load('words_list_lstm.npy')
words_embedding = np.load('word_embedding_lstm.npy')








class SentimentClassifier():

    def __init__(self,hdim,labels):

        tf.reset_default_graph()




        #placeholders
        input_x=tf.placeholder(tf.int32,shape=[None,None])
        output=tf.placeholder(tf.int32,shape=[None,])


        self.placeholders={'input':input_x,'output':output}

        word_embedding=tf.get_variable('word_embedding',shape=[400000,100],dtype=tf.float32,initializer=tf.constant_initializer(np.array(words_embedding)),trainable=False)

        embedding_lookup = tf.nn.embedding_lookup(word_embedding,input_x)

        sequence_len=tf.count_nonzero(input_x,axis=-1)

        with tf.variable_scope('encoder') as scope:

            cell=rnn.LSTMCell(num_units=hdim)

            model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=embedding_lookup,sequence_length=sequence_len,dtype=tf.float32)

        model_output,(fs,fc)=model

        weights_ = tf.get_variable('weights_a',shape=[2*hdim,labels],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        bias_    = tf.get_variable('bias_a',shape=[labels,],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        concat_output = tf.concat([fs.c,fc.c],axis=-1)

        # concat_output = tf.concat((tf.transpose(model_output[0],[1,0,2])[0],tf.transpose(model_output[1],[1,0,2])[0]))

        logits_out = tf.matmul(concat_output,weights_)+bias_

        #Normalization
        prob=tf.nn.softmax(logits_out)
        pred=tf.argmax(prob,axis=-1)


        #cross_entropy

        ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out,labels=output)
        loss = tf.reduce_mean(ce)

        #accuracy

        accuracy= tf.reduce_mean(tf.cast((tf.equal(tf.cast(pred,tf.int32),output)),tf.float32))

        #train
        training_ = tf.train.AdamOptimizer().minimize(loss)


        self.out ={  'logits': logits_out,
                     'prob'  : prob,
                     'pred'  : pred,
                     'loss'  : loss,
                     'accuracy': accuracy,
                     'training' :training_,

                     }


        self.testing = { 'embedding':embedding_lookup,   #8x10x100
                         'model_output' : model_output,  #8x10x12
                         'each_cell_output' : fs,        #8x12
                         'concat' : concat_output,       #8x24
                         'logits_out' : logits_out,      #8x10


        }





        self.for_testing = { 'logits': logits_out,
                            'prob'  : prob,
                            'pred'  : pred,
                            'model' : model



        }


def check(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(model.testing,feed_dict={model.placeholders['input']:np.random.randint(0,9,[8,10]),model.placeholders['output']:np.random.randint(0,9,[8,])})





if __name__=='__main__':
    model=SentimentClassifier(12,10)
    out=check(model)

    print(out['embedding'].shape,out['model_output'][0].shape,out['each_cell_output'].c.shape,out['concat'].shape,out['logits_out'].shape)






