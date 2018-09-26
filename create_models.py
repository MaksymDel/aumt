from allennlp.common import Params

from models import *


def create_model(X_vocab, Y_vocab, opts):
    """
    Builds the generators, discriminators, and embeddings.

    """
    ###################
    # CREATE EMBEDDINGS
    ###################

    Y_emb_params = Params({"vocab_namespace": "tokens",
                           "embedding_dim": opts.embedding_dim,
                           "pretrained_file": None,
                           "trainable": True
                           })

    X_emb_params = Y_emb_params.duplicate()

    Y_embedding = Embedding.from_params(vocab=Y_vocab, params=Y_emb_params)
    X_embedding = Embedding.from_params(vocab=X_vocab, params=X_emb_params)

    ###################
    # CREATE GENERATORS
    ###################

    seq2seq_lstm_params_Y = Params({
        "type": "lstm",
        "num_layers": opts.g_num_layers,
        "bidirectional": opts.g_bidirectional,
        "input_size": opts.embedding_dim,
        "hidden_size": opts.g_hidden_size
    })

    seq2seq_lstm_params_X = seq2seq_lstm_params_Y.duplicate()

    G_XtoY = VanillaRnn2Rnn(source_vocab=X_vocab,
                            target_vocab=Y_vocab,
                            source_embedding=X_embedding,
                            target_embedding=Y_embedding,
                            encoder=Seq2SeqEncoder.from_params(params=seq2seq_lstm_params_X),
                            max_decoding_steps=opts.g_max_decoding_steps)

    G_YtoX = VanillaRnn2Rnn(source_vocab=Y_vocab,
                            target_vocab=X_vocab,
                            source_embedding=Y_embedding,
                            target_embedding=X_embedding,
                            encoder=Seq2SeqEncoder.from_params(params=seq2seq_lstm_params_Y),
                            max_decoding_steps=opts.g_max_decoding_steps)

    #######################
    # CREATE DISCRIMINATORS
    #######################

    classifier_params_Y = Params({
        "type": "lstm",
        "bidirectional": opts.d_bidirectional,
        "num_layers": opts.d_num_layers,
        "input_size": opts.embedding_dim,
        "hidden_size": opts.d_hidden_size
    })

    classifier_params_X = classifier_params_Y.duplicate()

    D_Y = Seq2Binary(vocab=Y_vocab, embedding=Y_embedding,
                     seq2vec_encoder=Seq2VecEncoder.from_params(classifier_params_Y))
    D_X = Seq2Binary(vocab=X_vocab, embedding=X_embedding,
                     seq2vec_encoder=Seq2VecEncoder.from_params(classifier_params_X))

    #######################

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')
    else:
        print('Models are kept on CPU.')

    return G_XtoY, G_YtoX, D_X, D_Y, X_embedding, Y_embedding


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
    print("---------------------------------------")
