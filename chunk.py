import perc
import sys, optparse, os
from collections import defaultdict

def create_featureSchema(feat_list, labeled_list):
    '''
    function that returns a 2d dictionary containing the feature schema for all words in the input sentence
    first level key for dictionary is word_index for a sentence
    second level key is feature_name (as stated in feature schama)
    '''
    schema = {}

    # z variable used as counter to access and append features schema for each word
    z = 0
    feat_length = len(feat_list)/len(labeled_list)
    for x in range(0,len(labeled_list)):
        for y in range(0,feat_length):
            if y == 20 - 1:
                key = feat_list[y + z]
                value = labeled_list[x].split(' ')[2]
                schema[x].update({key : value})
            else:
                key = feat_list[y + z].split(':')[0]
                value = feat_list[y + z].split(':')[1]
                if not schema.get(x):
                    schema[x] = {key : value}
                else:
                    schema[x].update({key : value})
        z += feat_length

    return schema


def update_featVector(output_label, prev_output_label, true_features, true_label, prev_true_label, feat_vec):
    '''
    function that updates the feature vector by adding or subtracting one from the feature function
    If feature function is not in present in dictionary create key and set to -1 or +1
    If feature function is found, increment or decrement
    '''
    for k,v in true_features.iteritems():
        if k == 'B':
            output_featureFunction = (k + ':' + prev_output_label, output_label)
            true_featureFunction = (k + ':' + prev_output_label, true_label)
        else:
            output_featureFunction = (k + ':' + v, output_label)
            true_featureFunction = (k + ':' + v, true_label)

        if not feat_vec.get(output_featureFunction):
            feat_vec[output_featureFunction] = -1
        else:
            feat_vec[output_featureFunction] -= 1

        if not feat_vec.get(true_featureFunction):
            feat_vec[true_featureFunction] = 1
        else:
            feat_vec[true_featureFunction] += 1


def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    print 'numepochs = ', numepochs
    for count in range(0,numepochs):
        print "Epoch: " + str(count)
        for x in range(0,len(train_data)):
            true_features = create_featureSchema(train_data[x][1], train_data[x][0])
            output_label = perc.perc_test(feat_vec, train_data[x][0], train_data[x][1], tagset, tagset[0])
            for y in range(0,len(output_label)):
                true_label = train_data[x][0][y].split(' ')[2]
                if output_label[y] != true_label:
                    previous_true_label = train_data[x][0][y-1].split(' ')[2]
                    update_featVector(output_label[y], output_label[y-1], true_features[y], true_label, previous_true_label, feat_vec)

    return feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."

    # Each epoch (iteration) takes ~3 minutes. Don't know if this is too slow?
    # I ran the default 10 iterations which took just over 30 minutes.  Not sure if less iterations would result in a lower score
    print >>sys.stderr, "training ..."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)
