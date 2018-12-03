from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np

class RNNModel:
    """Wraps a Keras RNN model."""

    def __init__(self, recurrent_nodes=[25], dense_nodes=[5, 25], n_labels=2, n_features=0, sequence_length=20, dropout=0.2):
        """
        recurrent_nodes: a list of numbers of LSTM nodes to use in each layer
        dense_nodes: a list of numbers of dense nodes to use in each layer after
                     the RNN layers
        n_labels: the number of final classification labels
        n_features: the number of features in each input datapoint
        sequence_length: the length of sequence to use in training
        dropout: the proportion to use for dropout regularization
        """
        self.recurrent_nodes = recurrent_nodes
        self.dense_nodes = [size for size in dense_nodes]
        self.n_labels = n_labels
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.dropout = dropout

    def create(self):
        """Creates and compiles the RNN model according to initialization parameters."""
        self.model = Sequential()
        # TODO: dropout regularization
        # TODO: potential word2vec embedding
        for i, rec in enumerate(self.recurrent_nodes):
            if i == 0:
                self.model.add(LSTM(rec, input_shape=(self.n_features,self.sequence_length)))
            else:
                self.model.add(LSTM(rec))
        for size in self.dense_nodes:
            self.model.add(Dense(size, activation='relu'))
        self.model.add(Dense(self.n_labels, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, Xs, Ys, epochs=10, balance=True):
        """
        Trains the model using the given X input data and labels Y.

        Xs: matrix of size m x k x n, where m is the number of training examples,
            k is the number of features and n is the sequence length
        Ys: matrix of size m x l x n, where l is the number of labels
        epochs: number of epochs to train for
        balance: whether to balance the Xs and Ys to have an equal number of
            each Y label
        """
        if balance:
            new_X, new_Y = self.balance(Xs, Ys)
        else:
            new_X, new_Y = X, Y
        self.model.fit(new_X, new_Y, epochs=epochs, batch_size=100)

    def predict(self, X):
        """
        Produces a vector of state assignments given the X matrix (should be
        shape m x k x n - see train() for more information).
        """
        Y_one_hot = self.model.predict(X)
        return np.argmax(Y_one_hot, axis=1)

    def balance(self, X, Y):
        """
        Balances the given X and Y matrices so that there is an equal number of
        each Y label.
        """
        vals, counts = np.unique(Y, axis=0, return_counts=True)
        min_count = np.min(counts)
        num_rows = min_count * len(vals)
        new_X = np.zeros((num_rows, X.shape[1], X.shape[2]))
        new_Y = np.zeros((num_rows, Y.shape[1]))
        for i in range(len(vals)):
            indexes = np.random.choice(np.argwhere(Y[:,i] == 1).flatten(), size=min_count, replace=False)
            new_X[i * min_count: (i + 1) * min_count] = X[indexes]
            new_Y[i * min_count: (i + 1) * min_count] = Y[indexes]
        return new_X, new_Y


if __name__ == '__main__':
    import models
    from Bio import SeqIO
    import pickle
    import os
    # Test the RNNModel
    model = RNNModel()
    base_path = '/Users/venkatesh-sivaraman/Documents/School/MIT/6-047/proj/hicmotif/data/'
    with open(os.path.join(base_path, 'epigenomic_tracks', 'GM12878.pickle'), 'rb') as file:
        id, data = pickle.load(file)
    seq = None

    for record in SeqIO.parse(os.path.join(base_path, 'loop_sequences_GM12878.fasta'), 'fasta'):
        print(record.id.replace('chr', ''), id)
        if record.id.replace('chr', '') != id: continue
        seq = models.Sequence(id, str(record.seq), data)
        break

    print(len(seq.seq))
