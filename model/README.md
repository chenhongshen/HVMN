### Description
This repository hosts Hierarchical Variational Memory Network for Dialogue Generation (HVMN) model (VHRED) for dialouge generation inherited from the Serban et al. (2016a) and Serban et al. (2016c).



### Truncated BPTT
Both models are implemented using Truncated Backpropagation Through Time (Truncated BPTT).
The truncated computation is carried out by splitting each document (dialogue) into shorter sequences (e.g. 80 tokens) and computing gradients for each sequence separately, such that the hidden state of the RNNs on each subsequence are initialized from the preceding sequences (i.e. the hidden states have been forward propagated through the previous states).



### Creating Datasets
The script convert-text2dict.py can be used to generate model datasets based on text files with dialogues.
It only requires that the document contains end-of-utterance tokens &lt;/s&gt; which are used to construct the model graph, since the utterance encoder is only connected to the dialogue encoder at the end of each utterance.

Prepare your dataset as a text file for with one document per line (e.g. one dialogue per line). The documents are assumed to be tokenized. If you have validation and test sets, they must satisfy the same requirements.

Once you're ready, you can create the model dataset files by running:

python convert-text2dict.py &lt;training_file&gt; --cutoff &lt;vocabulary_size&gt; Training
python convert-text2dict.py &lt;validation_file&gt; --dict=Training.dict.pkl Validation
python convert-text2dict.py &lt;test_file&gt; --dict=Training.dict.pkl &lt;vocabulary_size&gt; Test

where &lt;training_file&gt;, &lt;validation_file&gt; and &lt;test_file&gt; are the training, validation and test files, and &lt;vocabulary_size&gt; is the number of tokens that you want to train on (all other tokens, but the most frequent &lt;vocabulary_size&gt; tokens, will be converted to &lt;unk&gt; symbols).

NOTE: The script automatically adds the following special tokens specific to movie scripts:
- end-of-utterance: &lt;/s&gt;
- end-of-dialogue: &lt;/d&gt;
- first speaker: &lt;first_speaker&gt;
- second speaker: &lt;second_speaker&gt;
- third speaker: &lt;third_speaker&gt;
- minor speaker: &lt;minor_speaker&gt;
- voice over: &lt;voice_over&gt;
- off screen: &lt;off_screen&gt;
- pause: &lt;pause&gt;

If these do not exist in your dataset, you can safely ignore these. The model will learn to assign approximately zero probability mass to them.



### Model Training
If you have Theano with GPU installed (bleeding edge version), you can train the model as follows:
1) Clone the Github repository
2) Create a new "Output" and "Data" directories inside it.
3) Unpack your dataset files into "Data" directory.
4) Create a new prototype inside state.py (look at prototype_ubuntu_HRED for an example)
5) From the terminal, cd into the code directory and run:

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py --prototype <prototype_name> > Model_Output.txt

where &lt;prototype_name&gt; is a state (model architecture) defined inside state.py.
Training a model to convergence on a modern GPU on the Ubuntu Dialogue Corpus with 46 million tokens takes about 1-2 weeks. If your GPU runs out of memory, you can adjust the bs (batch size) parameter in the model state, but training will be slower. You can also play around with the other parameters inside state.py.

(CURRENTLY NOT SUPPORTED) To test a model w.r.t. word perplexity run:

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python evaluate.py <model_name> Model_Evaluation.txt

where &lt;model_name&gt; is the model name automatically generated during training.



### Model Sampling & Testing

To generate model responses using beam search run:

    THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python sample.py <model_name> <contexts> <model_outputs> --beam_search --n-samples=<beams> --ignore-unk --verbose

where &lt;model_name&gt; is the name automatically generated during training, &lt;contexts&gt; is a file containing the dialogue contexts with one dialogue per line, and &lt;beams&gt; is the size of the beam search. The results are saved in the file &lt;model_outputs&gt;.

To compute the embedding-based metrics on the generated responses run:

    python Evaluation/embedding_metrics.py <ground_truth_responses> <model_outputs> <word_emb> 

where &lt;ground_truth_responses&gt; is a file containing the ground truth responses, &lt;model_outputs&gt; is the file generated above and &lt;word_emb&gt; is the path to the binarized word embeddings. For the word embeddings, we recommend to use Word2Vec trained on the GoogleNews Corpus: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM.