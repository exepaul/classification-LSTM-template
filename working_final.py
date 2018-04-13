import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tqdm import tqdm


validation_data=['Legal Sparring Continues in Bitcoin User s Battle with IRS Tax Sweep  http  ift tt number iTYYDC http  ift tt number iWnLed','MISSED BITCOIN DONT MISS THIS PR REG NOW legal block chain DO NOT MISS THIS http  tcpros co number XhMx crypto blockchain money euro','you are bad','you are not good','Well that dip under $1000 lasted about three years longer than I expected','Legal Sparring Continues in Bitcoin User’s Battle with IRS Tax Sweep','dip down under water','MISSED BITCOIN DONT MISS THIS PR REG NOW legal block chain DO NOT MISS THIS http  tcpros co number XhMx crypto blockchain money euro']
words = np.load('words_list_lstm.npy')
word_embedding = np.load('word_embedding_lstm.npy')

tweet_dataset=np.load('sentiment_final.npy')
epoch=50
batch_size=3
iteration=int(len(tweet_dataset)//batch_size)


labels_datr=[1,-1,0]
           # 0  1  2
labels2index={j:i for i,j in enumerate(labels_datr)}

def padding_data(data_):
    input_x_data = []
    max_value = max([len(i) for i in data_])

    final_data = [i + [0] * (max_value - len(i)) if len(i) < max_value else i for i in data_]


    return {'input': final_data}

class LSTMclassifier():
    def __init__(self, hdim, lables):
        tf.reset_default_graph()

        # placeholders
        input_x = tf.placeholder(tf.int32, shape=[None, None])
        output_y = tf.placeholder(tf.int32, shape=[None, ])

        self.placeholder = {'input': input_x, 'output': output_y}

        # word_embedding
        word_embedd = tf.get_variable('embedding', shape=[400000, 100], dtype=tf.float32,
                                      initializer=tf.constant_initializer(np.array(word_embedding)), trainable=False)
        embedding_lookup = tf.nn.embedding_lookup(word_embedd, input_x)

        # sequence_length
        sequence_le = tf.count_nonzero(input_x, axis=-1)

        # model
        with tf.variable_scope('encoder') as scope:
            cell = rnn.LSTMCell(num_units=hdim)
            model = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=embedding_lookup, sequence_length=sequence_le,
                                                    dtype=tf.float32)

        final_output, (fs, fc) = model

        # transform_output
        final_output_forward = tf.transpose(final_output[0], [1, 0, 2])
        final_output_backward = tf.transpose(final_output[1], [1, 0, 2])

        state_output = tf.concat([fs.c, fc.c], axis=-1)
        final_output_both = tf.concat([final_output_forward[0], final_output_backward[0]], axis=-1)

        # weights and fully_connected_layer
        weights = tf.get_variable('weights', shape=[2 * hdim, lables], dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-0.01, 0.01))

        bias = tf.get_variable('bias', shape=[lables], dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-0.01, 0.01))

        # logits ( final_output_matrix )

        logits_ = tf.matmul(final_output_both, weights) + bias

        # normalization
        prob = tf.nn.softmax(logits_)
        pred = tf.argmax(prob, axis=-1)

        # cross_entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=output_y)
        loss = tf.reduce_mean(ce)

        # accuracy
        accuracy = tf.reduce_mean(
            tf.cast(
                (tf.equal(
                    tf.cast(pred, tf.int32), output_y)),
                tf.float32))

        # training
        training_ = tf.train.AdamOptimizer().minimize(loss)

        self.out = {'logits': logits_,
                    'prob': prob,
                    'pred': pred,
                    'loss': loss,
                    'accuracy': accuracy,
                    'train': training_
                    }

        self.shape_checking = {'embedding_l': embedding_lookup,  # 2x4x100
                               'final_output': final_output,  # 2x4x10
                               'fs': fs,  # 2x10
                               'transpo': final_output_backward,  # 4x2x10
                               'for': final_output_forward,  # 4x2x10
                               'state': state_output,  # 2x20
                               'fina': final_output_both,  # 2x20
                               'weig': weights,  # 20x10
                               'log': logits_}  # 2x10

        self.interaction ={'model':model,
                           'prob':prob,
                           'pred':pred,
                           'logits__':logits_




        }


# checking model with dummy data


def rand_exe(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in tqdm(range(iteration)):
                batch = tweet_dataset[j * batch_size:(j + 1) * batch_size]
                tweets_data = np.array(padding_data([aa for aa, bb in batch])['input'])
                labels_data = np.array([labels_datr.index(bb) for aa, bb in batch])


                out_a=sess.run(model.out, feed_dict={model.placeholder['input']: tweets_data,
                                                         model.placeholder['output']: labels_data})
                print("epoch {} iteration {} loss {} accuracy {} ".format(i,j,out_a['loss'],out_a['accuracy']))


        for ow in validation_data:
            dataaa=ow.split()
            final_data = []
            for i in dataaa:
                try:
                    final_data.append(np.array(words).tolist().index(i.lower()))
                except ValueError:
                    final_data.append(399999)
            output_realwe = sess.run(model.interaction, feed_dict={model.placeholder['input']: [final_data]})

            print(ow)

            print(output_realwe['prob'],output_realwe['pred'])


        while True:

            user_input = str(input())
            data_sp = user_input.split()
            final_data = []
            for i in data_sp:
                try:
                    final_data.append(np.array(words).tolist().index(i.lower()))
                except ValueError:
                    final_data.append(399999)
            output_real = sess.run(model.interaction, feed_dict={model.placeholder['input']: [final_data]})

            print(output_real['prob'],output_real['pred'])


if __name__ == '__main__':
    model = LSTMclassifier(250, len(labels_datr))
    out = rand_exe(model)
    print(out['prob'],out['pred'])
