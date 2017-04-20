'''

Created on 22 Apr 2016

@author: af
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import pdb
import numpy as np
import sys
from os import path
import scipy as sp
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
import argparse
from lasagne.layers import DenseLayer, DropoutLayer
import logging
import json
import codecs
import pickle
from sklearn.decomposition import PCA
import gzip
from collections import OrderedDict, defaultdict, Counter
from haversine import haversine
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

    
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.random.seed(77)
random.seed(77)

    

#lasagne sparse layer for BoW text input
class SparseInputDenseLayer(DenseLayer):
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        activation = S.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
#lasagne sparse dropout layer for BoW text input
class SparseInputDropoutLayer(DropoutLayer):
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)



def load_data(data_dir, dataset):
    logging.info('loading data...')
    with open(path.join(data_dir, dataset + '.pkl'), 'rb') as fout:
        dataset = pickle.load(fout)
    return dataset
    



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def get_dare_vocab_vectorize(data_dir, vectorizer):
    texts = []
    vocabs = vectorizer.get_feature_names()
    vocabset = set(vocabs)
    wordfreqs = read_1m_words()
    topwords = [wordfreq[0] for wordfreq in wordfreqs]
    freqwords = set(topwords[0:10000])
    vocabs = [v for v in vocabs if v not in freqwords and v[0]!='#']
    texts = texts + vocabs
    #now read dare json files
    json_file = path.join(data_dir, 'geodare.cleansed.filtered.json')
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            if word in vocabset:
                texts.append(word)
            if subregions:
                texts.append(dialect + ' ' + subregions.lower())
            else:
                texts.append(dialect)
    texts = sorted(list(set([text.strip() for text in texts if len(text)>1])))
    return texts


def read_1m_words(input_file='./data/count_1w.txt'):
    wordcount = []
    with codecs.open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            word, count = line.strip().split('\t')
            wordcount.append((word, count))
    return wordcount

def nearest_neighbours(vocab, embs, k, model):
    #now read dare json files
    json_file = './data/geodare.cleansed.filtered.json'
    json_objs = []
    texts = []
    vocabset = set(vocab)
    dialect_words = defaultdict(list)
    dialect_subregions = {}
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            texts.append(word)
            if word in vocabset:
                dialect_words[dialect].append(word)
            if subregions:
                texts.append(subregions.lower())
                subregion_items = [item for item in subregions.lower().split(',') if len(item.strip()) > 0]
                dialect_subregions[dialect] = subregion_items
                texts.extend(subregion_items)
            else:
                texts.append(dialect)

    
    logging.info('creating dialect embeddings by multiplying subregion embdeddings')
    dialect_embs = OrderedDict()
    for dialect in sorted(dialect_words):
        dialect_items = dialect_subregions.get(dialect, [dialect])
        extended_dialect_items = []
        for dialect_item in dialect_items:
            itemsplit = dialect_item.split()
            extended_dialect_items.extend(itemsplit)
        itemsplit = dialect.split()
        extended_dialect_items.extend(itemsplit)
        dialect_item_indices = [vocab.index(item) for item in extended_dialect_items if item in vocabset]
        if len(dialect_item_indices) > 0:
            dialect_emb = np.zeros((1, embs.shape[1]))
            for _index in dialect_item_indices:
                dialect_emb = dialect_emb + embs[_index, :].reshape((1, embs.shape[1]))
            if len(dialect_item_indices) > 0:
                dialect_emb = dialect_emb / len(dialect_item_indices)
    
            dialect_embs[dialect] = dialect_emb
    target_X = np.vstack(tuple(dialect_embs.values()))
    #logging.info('MinMax Scaling each dimension to fit between 0,1')
    #target_X = scaler.fit_transform(target_X)
    #logging.info('l1 normalizing embedding samples')
    #target_X = normalize(target_X, norm='l1', axis=1, copy=False)

    #target_indices = np.asarray(text_index.values())
    #target_X = embs[target_indices, :]
    logging.info('computing nearest neighbours of dialects')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=10).fit(embs)
    distances, indices = nbrs.kneighbors(target_X)
    word_nbrs = [(dialect_embs.keys()[i], vocab[indices[i, j]]) for i in range(target_X.shape[0]) for j in range(k)]
    word_neighbours = defaultdict(list)
    for word_nbr in word_nbrs:
        word, nbr = word_nbr
        word_neighbours[word].append(nbr)
    
    return word_neighbours


def pca_dare_vocab(vocab, embs):
    #now read dare json files
    json_file = './data/geodare.cleansed.filtered.json'
    dialect_words = defaultdict(list)
    vocabset = set(vocab)
    all_words = set()
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')      
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            if word in vocabset:
                dialect_words[dialect].append(word)
                all_words.add(word)
    logging.info('the number of dare words covered is %d' % len(all_words))    

            
    word_embs = OrderedDict()
    for w in all_words:
        index = vocab.index(w)
        word_embs[w] = embs[index, :].reshape((1, embs.shape[1]))
    pca = PCA(n_components=2)
    words = word_embs.keys()
    dare_embs = np.vstack(tuple(word_embs.values()))
    dare_embs = pca.fit_transform(dare_embs)
    dialect_count = Counter()
    for dialect in sorted(dialect_words):
        _words = dialect_words[dialect]
        dialect_count[dialect] = len(_words)
        #logging.info('dialect %s has %d covered words.' %(dialect, len(_words)))
    two_dialects = ['southwest', 'new england']
    #two_dialects = [d[0] for d in dialect_count.most_common(2)]
    logging.info('drawing for %s' %str(two_dialects))
    pacific_northwest_words = dialect_words[two_dialects[0]]
    pacific_northwest_indices = [words.index(w) for w in pacific_northwest_words if w in all_words] 
    southeast_words = dialect_words[two_dialects[1]]
    southeast_indices = [words.index(w) for w in southeast_words if w in all_words]
    f, ax = plt.subplots()
    plt.axis('off')
    for i in pacific_northwest_indices:
        txt = words[i]
        plt.annotate(txt, (dare_embs[i, 0],dare_embs[i, 1]), fontsize=10, color='royalblue')
    for i in southeast_indices:
        txt = words[i]
        plt.annotate(txt, (dare_embs[i, 0],dare_embs[i, 1]), fontsize=10, color='crimson')
    plt.savefig('wordmap.pdf')
    


def nearest_neighbours_vectorize(vocab, embs, k, model):
    #now read dare json files
    json_file = './data/geodare.cleansed.filtered.json'
    dialect_words = defaultdict(list)
    dialect_subregions = {}
    vocabset = set(vocab)
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')       
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            if word in vocabset:
                if subregions:
                    dialect_words[dialect + ' ' + subregions.lower()].append(word)
                else:
                    dialect_words[dialect].append(word)

    
    logging.info('creating dialect embeddings for %d dialects' % len(dialect_words) )
    dialect_embs = OrderedDict()
    
    for dialect in sorted(dialect_words):
        if dialect in vocabset:
            index = vocab.index(dialect)
            dialect_embs[dialect] = embs[index, :].reshape((1, embs.shape[1]))
    target_X = np.vstack(tuple(dialect_embs.values()))
    logging.info('computing nearest neighbours of dialects')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=10).fit(embs)
    distances, indices = nbrs.kneighbors(target_X)
    word_nbrs = [(dialect_embs.keys()[i], vocab[indices[i, j]]) for i in range(target_X.shape[0]) for j in range(k)]
    word_neighbours = defaultdict(list)
    for word_nbr in word_nbrs:
        word, nbr = word_nbr
        word_neighbours[word].append(nbr)
    
    return word_neighbours


def calc_recall(vocab, word_nbrs, k, freqwords=set(), model='mlp'):
    json_file = './data/geodare.cleansed.filtered.json'
    json_objs = []
    texts = []
    vocabset = set(vocab)
    dialect_words = defaultdict(list)
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            if model == 'word2vec':
                if word in vocabset:
                    dialect_words[dialect].append(word)
            else:
                if word in vocabset:
                    if subregions:
                        dialect_words[dialect + ' ' + subregions.lower()].append(word)
                    else:
                        dialect_words[dialect].append(word)
            

            
    recalls = []
    info = []
    total_true_positive = 0
    total_positive = 0
    for dialect, nbrs in word_nbrs.iteritems():
        dialect_has = 0
        dialect_total = 0
        nbrs = [nbr for nbr in nbrs if nbr not in freqwords and nbr[0]!='#']
        nbrs = set(nbrs[0:k])
        if dialect in dialect_words:
            dwords = set(dialect_words[dialect])
            dialect_total = len(dwords)
            total_positive += dialect_total
            if dialect_total == 0:
                logging.info('zero dialect words ' + dialect)
                continue
            for dword in dwords:
                if dword in nbrs:
                    #logging.info('word %s found in dialect %s within %d results' %(dword, dialect, k))
                    dialect_has += 1
                    total_true_positive += 1
            recall = 100 * float(dialect_has) / dialect_total
            recalls.append(recall)
            info.append((dialect, dialect_total, recall))
        else:
            logging.info('this dialect does not exist: ' + dialect)
    logging.info('recall at ' + str(k))
    logging.info('#relevant %d, #hits %d' % (total_positive, total_true_positive))
    sum_support = sum([inf[1] for inf in info])
    logging.info('micro recall :' + str(float(total_true_positive) * 100 / total_positive))





def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)
        
    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    return np.mean(distances), np.median(distances), acc_at_161

def geo_mlp(data_dir, dataset, n_epochs=10, batch_size=1000, regul_coefs=[5e-5, 5e-5],  hidden_layer_size=None, drop_out_coefs=[0.5, 0.5]):
    '''
    Run the MLP geolocation model, evaluation both geolocation and dialect term detection
    '''
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, userLocation, classLatMedian, classLonMedian, vectorizer = load_data(data_dir, dataset)
    logging.info('building the network...')
    in_size = X_train.shape[1]
    best_params = None
    best_dev_acc = 0.0
    drop_out_hid, drop_out_in = drop_out_coefs
    out_size = len(set(Y_train.tolist()))
    
    logging.info('hidden layer size is ' + str(hidden_layer_size))
    X_sym = S.csr_matrix(name='inputs', dtype='float32')
    y_sym = T.ivector()    
    
    l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                     input_var=X_sym)
    l_in = lasagne.layers.dropout(l_in, p=drop_out_in)

    l_hid1 = SparseInputDenseLayer(
        l_in, num_units=hidden_layer_size,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
        
    l_out = lasagne.layers.DenseLayer(
        l_hid1, num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)


    embedding = lasagne.layers.get_output(l_hid1, X_sym, deterministic=True)
    output = lasagne.layers.get_output(l_out, X_sym, deterministic=True)
    pred = output.argmax(-1)
    loss = lasagne.objectives.categorical_crossentropy(output, y_sym)
    loss = loss.mean()

    l1_share_out = 0.5
    l1_share_hid = 0.5
    regul_coef_out, regul_coef_hid = regul_coefs
    logging.info('regul coefficient for output and hidden layers are ' + str(regul_coefs))
    l1_penalty = lasagne.regularization.regularize_layer_params(l_out, l1) * regul_coef_out * l1_share_out
    l2_penalty = lasagne.regularization.regularize_layer_params(l_out, l2) * regul_coef_out * (1-l1_share_out)
    l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef_hid * l1_share_hid
    l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef_hid * (1-l1_share_hid)
    loss = loss + l1_penalty + l2_penalty


    acc = T.mean(T.eq(pred, y_sym))

    parameters = lasagne.layers.get_all_params(l_out, trainable=True)
    #updates = lasagne.updates.adam(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    updates = lasagne.updates.adamax(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
    f_get_embeddings = theano.function([X_sym], embedding)
    f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
    f_val = theano.function([X_sym, y_sym], [loss, acc])
    f_predict = theano.function([X_sym], pred)
    f_predict_proba = theano.function([X_sym], output)
    
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_dev = X_dev.astype('float32')


    Y_train = Y_train.astype('int32')
    Y_test = Y_test.astype('int32')
    Y_dev = Y_dev.astype('int32')
    model_file = path.join(data_dir, dataset + '.model.pkl')
    if path.exists(model_file):
        logging.info('loading stored parameteres...')
        with open(model_file, 'rb') as fin:
            best_params = pickle.load(fin)
    else:
        logging.info('training (n_epochs, batch_size) = (' + str(n_epochs) + ', ' + str(batch_size) + ')' )
        n_validation_down = 0
        for n in xrange(n_epochs):
            for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
                x_batch, y_batch = batch
                l_train, acc_train = f_train(x_batch, y_batch)
                l_val, acc_val = f_val(X_dev, Y_dev)
            logging.info('dev results after epoch')
            mean, median, acc_at_161 = geo_eval(Y_dev, f_predict(X_dev), U_dev, classLatMedian, classLonMedian, userLocation)
            if acc_at_161 > best_dev_acc:
                best_dev_acc = acc_at_161
                best_params = lasagne.layers.get_all_param_values(l_out)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            l_val, acc_val = f_val(X_dev, Y_dev)
            logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val) + ',best_val_acc ' + str(best_dev_acc))
            if n_validation_down > 3:
                logging.info('validation results went down. early stopping ...')
                break
        logging.info('storing best parameters...')
        with open(model_file, 'wb') as fout:
            pickle.dump(best_params, fout)
    #restore best validation model
    lasagne.layers.set_all_param_values(l_out, best_params)
    logging.info('***************** final results based on best validation model **************')
    logging.info('dev results')
    mean, median, acc_at_161 = geo_eval(Y_dev, f_predict(X_dev), U_dev, classLatMedian, classLonMedian, userLocation)
    logging.info('test results')
    mean, median, acc_at_161 = geo_eval(Y_test, f_predict(X_test), U_test, classLatMedian, classLonMedian, userLocation)

    if dataset != 'na':
        return

    logging.info('reading DARE vocab...')
    dare_vocab = get_dare_vocab_vectorize(data_dir, vectorizer)
    X_dare = vectorizer.transform(dare_vocab)
    X_dare = X_dare.astype('float32')
    logging.info('getting DARE embeddings...')
    X_dare_embs = f_get_embeddings(X_dare)
    logging.info('drawing wordmap...')
    #pca_dare_vocab(dare_vocab, X_dare_embs)
    dialect_eval(data_dir, vocab=dare_vocab, embs=X_dare_embs, model='mlp') 

def load_word2vec(fname):
    import gensim
    ''' load a pre-trained binary format word2vec into a dictionary
    the model is downloaded from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download'''
    word2vec = gensim.models.word2vec.Word2Vec.load_word2vec_format(fname, binary=True)
    return word2vec


def dialect_eval(data_dir, vocab, embs, model):
    vocab_size = len(vocab)
    logging.info('vocab size: %d'  % vocab_size)    
    percents = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    percents = [int(p* vocab_size) for p in percents]
    #adjust percents to match the vocabulary of mlp and lr
    percents = [129, 258, 517, 1292, 2585, 5170, 12952]
    if model == 'word2vec':
        word_nbrs = nearest_neighbours(vocab, embs, k=int(len(vocab)), model=model)
    else:
        word_nbrs = nearest_neighbours_vectorize(vocab, embs, k=20000, model=model)
    with open(path.join(data_dir, model + '_dialect_words.pkl'), 'wb') as fout:
        pickle.dump(word_nbrs, fout)
    wordfreqs = read_1m_words()
    topwords = [wordfreq[0] for wordfreq in wordfreqs]
    freqwords = set(topwords[0:10000])
    

    for r_at_k in percents:
        calc_recall(vocab, word_nbrs=word_nbrs, k=r_at_k, freqwords=freqwords, model=model)




def dialectology(data_dir='./data/', dataset='na', model='mlp'):
    if model == 'mlp':
        _coefs =  [1e-9, 1e-9]
        batch_size = 10000
        _hidden_layer_size = 256 * 8
        n_epochs = 20
        geo_mlp(data_dir, dataset, n_epochs=n_epochs, batch_size=batch_size, regul_coefs=_coefs, hidden_layer_size=_hidden_layer_size, drop_out_coefs=[0.01, 0.01])
    elif model == 'lr':
        with open(path.join(data_dir, 'lr_model.pkl'), 'rb') as fout:
            clf, vectorizer = pickle.load(fout)
        logging.info('reading DARE vocab...')
        #vectorizer_mlp = load_data(dataset)[-1]
        dare_vocab = get_dare_vocab_vectorize(data_dir, vectorizer)
        
        X_dare = vectorizer.transform(dare_vocab)
        logging.info('getting DARE embeddings...')
        lr_embeddings = clf.predict_proba(X_dare)
        dialect_eval(data_dir, vocab=dare_vocab, embs=lr_embeddings, model=model)
    elif model == 'word2vec':
        vectorizer = load_data(data_dir, dataset)[-1]
        logging.info('loading w2v embeddings...')
        logging.info('reading DARE vocab...')
        dare_vocab = get_dare_vocab_vectorize(data_dir, vectorizer)
        vocabset = set(dare_vocab)
        word2vec_model = load_word2vec('/home/arahimi/GoogleNews-vectors-negative300.bin.gz')
        w2v_vocab = [v for v in word2vec_model.vocab if v in vocabset]
        logging.info('vstacking word vectors into a single matrix for %d vocabs' %len(w2v_vocab))
        w2v_embs = np.vstack(tuple([np.asarray(word2vec_model[v]).reshape((1,300)) for v in w2v_vocab]))
        dialect_eval(data_dir, vocab=w2v_vocab, embs=w2v_embs, model=model)
                
def tune(data_dir, dataset, num_iter=50):
    #randomized search
    for _i in range(num_iter):
        random.seed()
        if dataset == 'cmu':
            hidden_layer_size = random.choice([32 * x for x in range(2, 50)])
            batch_size = 100
        elif dataset == 'na':
            hidden_layer_size = random.choice([256 * x for x in range(2, 20)])
            batch_size = 10000
        elif dataset == 'world':
            hidden_layer_size = hidden_layer_size = random.choice([256 * x for x in range(2, 20)])
            batch_size = 10000 
        coefs = random.choice([[x, x] for x in [1e-5, 5e-5, 5e-6, 1e-6, 1e-7] ])
        drop_out_ceofs = random.choice([[x, x] for x in [0.4, 0.5, 0.6] ])
        n_epochs = 20
        np.random.seed(77) 
        geo_mlp(data_dir, dataset, n_epochs=n_epochs, batch_size=batch_size, regul_coefs=coefs, hidden_layer_size=hidden_layer_size, drop_out_coefs=drop_out_ceofs)
        logging.info('#iter %d, coef %s, hidden %d, drop %s' %(_i, str(coefs), hidden_layer_size, str(drop_out_ceofs)))

def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset','--dataset', metavar='str', help='dataset name (cmu, na, world)', type=str, default='na')
    parser.add_argument('-model','--model', metavar='str', help='dialectology model (mlp, lr, word2vec)', type=str, default='mlp')
    parser.add_argument('-datadir', type=str, help='directory where input datasets (cmu.pkl, na.pkl, world.pkl) are located.')
    parser.add_argument('-tune', action='store_true', help='if true, tune the hyper-parameters.')
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args.tune:
        tune(args.datadir, args.dataset)
    else:
        if args.dataset == 'na':
            dialectology(args.datadir, dataset = args.dataset, model=args.model)
        elif args.dataset == 'cmu':   
            coefs = [1e-5, 1e-5]
            batch_size = 100
            hidden_layer_size = 32 * 30
            n_epochs = 20
            drop_out_coefs = [0.5, 0.5]
            geo_mlp(args.datadir, args.dataset, n_epochs=n_epochs, batch_size=batch_size, regul_coefs=coefs, hidden_layer_size=hidden_layer_size, drop_out_coefs=drop_out_coefs)
        elif args.dataset == 'world':
            coefs = [1e-6, 1e-6]
            batch_size = 10000
            hidden_layer_size = 930 * 4
            n_epochs = 20
            geo_mlp(args.datadir, args.dataset, n_epochs=n_epochs, batch_size=batch_size, regul_coefs=coefs, hidden_layer_size=hidden_layer_size, drop_out_coefs=[0.5, 0.5])
    




