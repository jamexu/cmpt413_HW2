### Readme for Assignment 2 - Phrasal Chunking

Program uses perc_train() skeleton from default.py

Takes inputs:
train_data[] - 2D list of all sentences in training data
tagset - The chunk tags list.  Default B-NP (most common)
numepochs - Number of epochs (iterations) to run

For each line (sentence), create a 2D dictionary with create_featureSchema()_ function.  Function takes parameters: train_data[i][0] and train_data[i][0]. First key of the 2D dictionary is the word index of the sentence.  Second key is the feature name described in feature schema (u01, u02 etc.).  Returns a 2D dictionary: true_features.

Create output_labels using viterbi algorithm perc_test(), provided in perc.py. Compare the output labels with the true labels.  If labels match then no need to update vector.  

If output label and true label don't match, then use update_featVector() to update the vector.  update_featvector() function takes as parameters: output_label, previous_output_label, true_label, previous_true_label, true_features[word_index], and feature_vector

If feature function already exists in feature_vector, then increment or decrement by 1 depending on the feature.  If feature function not found in vector, then create a new key for the feature function.

Repeat for numepochs.