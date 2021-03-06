from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
import numpy as np
import models
from Bio import SeqIO
import pickle
import os, sys
from main import load_data, generate_X
from keras.models import load_model

class RNNModel:
    """Wraps a Keras RNN model."""

    def __init__(self, recurrent_nodes=[25], dense_nodes=[25, 25], n_labels=2, n_features=0, sequence_length=20, dropout=0.2):
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
            if self.dropout > 0.0:
                self.model.add(Dropout(self.dropout))
        for size in self.dense_nodes:
            self.model.add(Dense(size, activation='relu'))
            if self.dropout > 0.0:
                self.model.add(Dropout(self.dropout))
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
            new_X, new_Y = Xs, Ys
        self.model.fit(new_X, new_Y, epochs=epochs, batch_size=160)

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

def load_training_data(path, n_labels, data, seq_length, spacing):
    """
    Loads pre-assigned state labels from a CSV file, associating the vectors with
    the appropriate interaction matrix. Returns an X and Y matrix.
    """
    X, ranges = generate_X(data, seq_length, spacing)
    range_mapping = {data[i][1].identifier: ranges[i] for i in range(len(data))}
    seen_ids = set()

    Y_single = np.zeros((X.shape[0],), dtype=int)
    with open(path, 'r') as file:
        for line in file:
            comps = line.strip().split(',')
            id = comps.pop(0)
            if id not in range_mapping: continue
            start, stop = range_mapping[id]
            score = comps.pop(0)
            asg = np.array([int(float(c)) for c in comps])
            #asg = asg[seq_length // spacing:] != asg[seq_length // spacing - 1:-1]
            assert asg.shape[0] >= stop - start, "incorrect length: {} vs {}-{} ({})".format(asg.shape[0], start, stop, stop - start)
            Y_single[start:stop] = asg[:stop - start]
            seen_ids.add(id)
            if len(seen_ids) == len(range_mapping): break

    #assert len(seen_ids) == len(range_mapping), "missing identifiers: " + str(set(range_mapping.keys()) - seen_ids)
    seen_ranges = np.array(sorted([x for id in seen_ids for x in range(*range_mapping[id])]))
    Y = np.zeros((X.shape[0], n_labels), dtype=int)
    for i in range(Y_single.shape[0]):
        Y[i,Y_single[i]] = 1
    return X[seen_ranges], Y[seen_ranges]

def train_big(base, data, seq_length, spacing, batch_size, n_epochs):
    """Trains an RNN using the full dataset and a predetermined architecture."""
    model = None
    #train_indexes = np.array([int(i) for i in open(base + "rnns_small_train_batch/train_indexes.txt").read().strip().split(",")]
    train_indexes = np.random.choice(len(data), size=(int(len(data) * 0.2),), replace=False)
    test_indexes = list(set(range(len(data))) - set(train_indexes))
    print("Train: {}".format(list(train_indexes)))
    #batch = [data[i] for i in train_indexes]
    #X, Y = load_training_data(base + "dp_assignments/assignments_{}.csv".format(pid), n_labels, batch, seq_length, spacing)

    for epoch in range(n_epochs):
        # Random batch of data for each epoch
        b_size = int(len(train_indexes) / 5)
        start_idx = (epoch % 5) * b_size
        indexes = train_indexes[start_idx:start_idx + b_size] #np.random.choice(len(train_indexes), size=(batch_size,), replace=False)
        batch = [data[i] for i in indexes]

        X, Y = load_training_data(base + "dp_assignments/assignments_{}.csv".format(pid), n_labels, batch, seq_length, spacing)
        if model is None:
            model = RNNModel(recurrent_nodes=[100], dense_nodes=[50, 50], n_labels=Y.shape[1], n_features=X.shape[1], sequence_length=X.shape[2])
            #model.model = load_model(base + "rnns_small_train_batch/iteration_20.hd5")
            model.create()

        print(X.shape, Y.shape)
        model.train(X, Y, epochs=1, balance=False)
        if epoch % 5 == 4:
            indexes = np.random.choice(len(test_indexes), size=(batch_size // 2,), replace=False)
            X_test, Y_test = load_training_data(base + "dp_assignments/assignments_{}.csv".format(pid), n_labels, [data[i] for i in indexes], seq_length, spacing)
            acc = np.sum(model.predict(X) == np.argmax(Y, axis=1)) / X.shape[0]
            print("Evaluation:", acc)

            dir = "rnns_predict_current_state"
            if not os.path.exists(base + dir): os.mkdir(base + dir)
            model.model.save(base + dir + "/iteration_{}.hd5".format(epoch))

def test_architectures(base, data, seq_length, spacing, batch_size, n_epochs):
    """Test various architectures for the RNN."""

    # Determine shape of input with a trial set
    X, Y = load_training_data(base + "dp_assignments/assignments_{}.csv".format(pid), n_labels, [data[0]], seq_length, spacing)
    kwargs = {'n_labels': Y.shape[1], 'n_features': X.shape[1], 'sequence_length': X.shape[2]}
    model_params = [
        {'recurrent_nodes': [50], 'dense_nodes': [25, 25]},
        {'recurrent_nodes': [100], 'dense_nodes': [25, 25]},
        {'recurrent_nodes': [200], 'dense_nodes': [25, 25]},
        {'recurrent_nodes': [100], 'dense_nodes': [50, 50]},
        {'recurrent_nodes': [200], 'dense_nodes': [50, 50]},
        {'recurrent_nodes': [100], 'dense_nodes': [100, 100]},
        {'recurrent_nodes': [200], 'dense_nodes': [100, 100]},
        {'recurrent_nodes': [100], 'dense_nodes': [100, 100, 100]},
        {'recurrent_nodes': [200], 'dense_nodes': [100, 100, 100]},
    ]

    train_indexes = np.random.choice(len(data), size=(int(len(data) * 0.75),), replace=False)
    test_indexes = list(set(range(len(data))) - set(train_indexes))
    print("{} training, {} test".format(len(train_indexes), len(test_indexes)))

    results = {}

    for i, model_p in enumerate(model_params):
        model_id = str(model_p)
        model_p.update(kwargs)
        model = RNNModel(**model_p)
        model.create()

        for epoch in range(n_epochs):
            # Random batch of data for each epoch
            indexes = np.random.choice(len(train_indexes), size=(batch_size,), replace=False)
            batch = [data[train_indexes[i]] for i in indexes]

            X, Y = load_training_data(base + "dp_assignments/assignments_{}.csv".format(pid), n_labels, batch, seq_length, spacing)
            print(X.shape, Y.shape)

            model.train(X, Y, epochs=1, balance=False)

        # Evaluate
        print("Evaluating...")
        indexes = np.random.choice(len(test_indexes), size=(batch_size * 2,), replace=False)
        batch = [data[i] for i in indexes]

        X, Y = load_training_data(base + "dp_assignments/assignments_{}.csv".format(pid), n_labels, batch, seq_length, spacing)
        print(X.shape, Y.shape)

        acc = np.sum(model.predict(X) == np.argmax(Y, axis=1)) / X.shape[0]
        results[model_id] = acc

    return results


if __name__ == '__main__':
    pid = sys.argv[1]
    n_labels = int(sys.argv[2])
    base = "../data/"
    data = load_data(base + "GM12878_10k", base + "loop_sequences_GM12878.fasta", base + "epigenomic_tracks/GM12878.pickle")
    seq_length = 100
    spacing = 50# if test else 10
    batch_size = 100
    n_epochs = 100
    train_big(base, data, seq_length, spacing, batch_size, n_epochs)
    #n_epochs = 50
    #results = test_architectures(base, data, seq_length, spacing, batch_size, n_epochs)
    #with open(base + "architectures_results.pickle", "wb") as file:
    #    pickle.dump(results, file)
