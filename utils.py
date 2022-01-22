import string
import pickle
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score


def getCharVocab():
    """
    Builds the mappings for character
    :return: num2char : dict made of int : char
    """
    chars = list(string.ascii_lowercase+string.digits+string.punctuation)
    char2num = {'<PAD>': 0, '<OOV>': 1}
    num2char = {0: '<PAD>', 1: '<OOV>'}
    for i in range(len(chars)):
        char2num[chars[i]] = 2 + i
        num2char[2 + i] = chars[i]
    return num2char, char2num


def getDataset():
    """
    Reads saved datasets
    :return: texts, labels, max_len_word
    """
    # Read
    with open("pickle_files/texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("pickle_files/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("pickle_files/mappings/max_len_word.pkl", "rb") as f:
        max_len_word = pickle.load(f)
    return texts, labels, max_len_word


def getMappings():
    """
    Reads saved mappings
    :return: [num2tag, tag2num, num2word, word2num]
    """
    with open("pickle_files/mappings/number2tag.pkl", "rb") as f:
        num2tag = pickle.load(f)
    with open("pickle_files/mappings/tag2number.pkl", "rb") as f:
        tag2num = pickle.load(f)
    with open("pickle_files/mappings/number2word.pkl", "rb") as f:
        num2word = pickle.load(f)
    with open("pickle_files/mappings/word2number.pkl", "rb") as f:
        word2num = pickle.load(f)

    return [num2tag, tag2num, num2word, word2num]


def getEmbeddingsLookup(emb_name, word2num):
    """
    Computes the embeddings lookup matrix and embedding matrix
    Embedding lookup matrix is a dictionary of word : word_embedding (vector)
    Embedding matrix is a matrix where each row is a vector corresponding to a word
    :param emb_name: name of the pretrained word embeddings
    :param word2num: mapping of the dataset words
    :return: embeddings lookup matrix and embeddings matrix
    """
    EMBEDDING_FILE = ''
    if emb_name == 'pubmed':
        EMBEDDING_FILE = 'pretrained_embeddings/PubMed-and-PMC-w2v.bin'
    elif emb_name == 'smth_else':
        # give another file
        EMBEDDING_FILE = 'pretrained_embeddings/'
    # Load vectors directly from the file
    wv_from_bin = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    vector_dim = wv_from_bin[list(wv_from_bin.vocab.keys())[0]].shape[0]
    embeddingsMatrix = [np.zeros(vector_dim), np.zeros(vector_dim)]
    embeddingsLookup = {'<PAD>': 0, '<OOV>': 1}
    for word in word2num.keys():
        if word in wv_from_bin.vocab:
            coeffs = np.asarray(wv_from_bin[word], dtype='float32')
            embeddingsMatrix.append(coeffs)
            embeddingsLookup[word] = len(embeddingsLookup)
    del wv_from_bin
    embeddingsMatrix = np.asarray(embeddingsMatrix)

    # Dump to file
    with open("pickle_files/mappings/embeddingsMatrix.pkl", "wb") as f:
        pickle.dump(embeddingsMatrix, f)
    with open("pickle_files/mappings/embeddingsLookup.pkl", "wb") as f:
        pickle.dump(embeddingsLookup, f)

    return embeddingsMatrix, embeddingsLookup


def loadEmbeddingLookup():
    with open("pickle_files/mappings/embeddingsMatrix.pkl", "rb") as f:
        embeddingsMatrix = pickle.load(f)
    with open("pickle_files/mappings/embeddingsLookup.pkl", "rb") as f:
        embeddingsLookup = pickle.load(f)
    return embeddingsMatrix, embeddingsLookup


def shuffle_and_split(input_seq, indices, VALIDATION_SPLIT=0.25):
    """
    Shuffle the dataset with given indices and split it
    :param input_seq: a dataset
    :param indices: a set of indices
    :param VALIDATION_SPLIT: portion of dataset given as validation
    :return: input_seq_train, input_seq_val
    """
    input_seq = input_seq[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * input_seq.shape[0])
    input_seq_train = input_seq[:-nb_validation_samples]
    input_seq_val = input_seq[-nb_validation_samples:]
    return input_seq_train, input_seq_val


class F1Metrics(Callback):
    """
    Class that compute the F1 Score. It works as Callback for keras
    """
    def __init__(self, id2label, pad_value=0, validation_data=None):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.labels = [label for label in id2label.values() if label not in ['o', '<oov>']]
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.is_fit = validation_data is None

    def find_pad_index(self, array):
        """Find padding index.
        Args:
            array (list): integer list.
        Returns:
            idx: padding index.
        Examples:
             >>> array = [1, 2, 0]
             >>> self.find_pad_index(array)
             2
        """
        try:
            return list(array).index(self.pad_value)
        except ValueError:
            return len(array)

    def get_length(self, y):
        """Get true length of y.
        Args:
            y (list): padded list.
        Returns:
            lens: true length of y.
        Examples:
            >>> y = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
            >>> self.get_length(y)
            [1, 2, 3]
        """
        #lens = [self.find_pad_index(row) for row in y]
        lens = []
        for row in y:
            lens.append(row[row != 0].shape[0])

        return lens

    def convert_idx_to_name(self, y, lens):
        """Convert label index to name.
        Args:
            y (list): label index list.
            lens (list): true length of y.
        Returns:
            y: label name list.
        Examples:
            >>> # assumes that id2label = {1: 'B-LOC', 2: 'I-LOC'}
            >>> y = [[1, 0, 0], [1, 2, 0], [1, 1, 1]]
            >>> lens = [1, 2, 3]
            >>> self.convert_idx_to_name(y, lens)
            [['B-LOC'], ['B-LOC', 'I-LOC'], ['B-LOC', 'B-LOC', 'B-LOC']]
        """
        y = [[self.id2label[idx] for idx in row[:l]]
             for row, l in zip(y, lens)]
        return y

    def predict(self, X, y):
        """Predict sequences.
        Args:
            X (list): input data.
            y (list): tags.
        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict_on_batch(X)

        lens = self.get_length(y)

        y_true = self.convert_idx_to_name(y, lens)
        y_pred = self.convert_idx_to_name(y_pred, lens)

        return y_true, y_pred

    def score(self, y_true, y_pred):
        """Calculate f1 score.
        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.
        Returns:
            score: f1 score.
        """
        score = flat_f1_score(y_true, y_pred, average='weighted', labels=self.labels)
        print(' - f1: {:04.2f}'.format(score * 100))

        report = flat_classification_report(y_pred=y_pred, y_true=y_true)
        print(report)
        return score

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_true, y_pred = self.predict(X, y)
        score = self.score(y_true, y_pred)
        logs['f1'] = score

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X, y in self.validation_data:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        score = self.score(y_true, y_pred)
        logs['f1'] = score


def plot_history(model_name, history):
    """
    Simple function to plot the training variables and save the figure
    :param model_name:
    :param history:
    :return:
    """
    loss_list = [s for s in history.history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.history.keys() if 'acc' in s and 'val' in s]
    val_f1_list = [s for s in history.history.history.keys() if 'f1' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(1, len(history.history.history[loss_list[0]]) + 1)
    colours_pairs = [('g', 'b'), ('r', 'c'), ('m', 'y'), ('k', 'w')]

    # Loss
    plt.figure(1)
    for l1, l2, color_pair in zip(loss_list, val_loss_list, colours_pairs):
        plt.plot(epochs, history.history.history[l1], color_pair[0],
                 label='Training ' + l1 + '(' + str(str(format(history.history.history[l1][-1], '.5f')) + ')'))
        plt.plot(epochs, history.history.history[l2], color_pair[1],
                 label='Validation ' + l2 + '(' + str(str(format(history.history.history[l2][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fn = "images/" + model_name + "_losses.jpg"
    plt.savefig(fn)

    # Accuracy
    plt.figure(2)
    for l1, l2, color_pair in zip(acc_list, val_acc_list, colours_pairs):
        plt.plot(epochs, history.history.history[l1], color_pair[0],
                 label='Training ' + l1 + '(' + str(str(format(history.history.history[l1][-1], '.5f')) + ')'))
        plt.plot(epochs, history.history.history[l2], color_pair[1],
                 label='Validation ' + l2 + '(' + str(str(format(history.history.history[l2][-1], '.5f')) + ')'))

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    fn = "images/" + model_name + "_accuracies.jpg"
    plt.savefig(fn)
    plt.show()

    # F1 score
    plt.figure(3)
    for l1 in val_f1_list:
        plt.plot(epochs, history.history.history[l1], colours_pairs[0][0],
                 label='Validation ' + l1 + '(' + str(str(format(history.history.history[l1][-1], '.5f')) + ')'))

    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    fn = "images/" + model_name + "_F1Score.jpg"
    plt.savefig(fn)
    plt.show()


