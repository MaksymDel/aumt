# Usage:
# ======
#############################
# python cycle_gan_mt.py -h
##############################

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np

# Local imports
import utils
from data_loader import get_iterator_vocab, MonolingualDatasetReader

from create_models import create_model

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def get_next_batch_mask(batch_iterator, embedding):
    sampled_batch = batch_iterator.__next__()
    sampled_indeces = sampled_batch['sentence']['tokens']
    sampled_indeces = utils.to_var(sampled_indeces).long()
    embedded_tokens = embedding.forward(sampled_indeces)
    mask = sampled_indeces != 0

    return embedded_tokens, mask


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.load, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.load, 'D_X.pkl')
    D_Y_path = os.path.join(opts.load, 'D_Y.pkl')

    G_XtoY = None #CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = None #CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = None #DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = None #DCDiscriminator(conv_dim=opts.d_conv_dim)

    # print(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage).keys()) #To REMOVE

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def save_samples(iteration, fixed_embedded_sentences_Y, fixed_mask_Y, fixed_embedded_sentences_X, fixed_mask_X, G_YtoX,
                 G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    _, _, predicted_tokens_X, predicted_tokens_softmax_X, _ = G_YtoX(fixed_embedded_sentences_Y, fixed_mask_Y).values()
    _, _, predicted_tokens_Y, predicted_tokens_softmax_X, _ = G_XtoY(fixed_embedded_sentences_X, fixed_mask_X).values()

    # X, fake_X = utils.to_data(fixed_X), utils.to_data(fake_X)
    # Y, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    path = os.path.join(opts.sample_dir, 'hyps-{:06d}.X'.format(iteration))
    with open(path, 'w') as f:
        [f.write(" ".join(_list) + '\n') for _list in predicted_tokens_X]
    print('[not supported] Saved {}'.format(path))
    print('/HYP X:/ ', " ".join(predicted_tokens_X[0]))

    path = os.path.join(opts.sample_dir, 'hyps-{:06d}.Y'.format(iteration))
    with open(path, 'w') as f:
        [f.write(" ".join(_list) + '\n') for _list in predicted_tokens_Y]
    print('[not supported] Saved {}'.format(path))
    print('/HYP Y:/ ', " ".join(predicted_tokens_Y[0]))


def loss_helper(x, m):
    return torch.sum((x - 1) ** 2) / m


def loss_helper1(x, m):
    return torch.sum(x ** 2) / m


def training_loop(batch_iterator_X, batch_iterator_Y,
                  dev_batch_iterator_X, dev_batch_iterator_Y,
                  vocab_X, vocab_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    if opts.load:
        G_XtoY, G_YtoX, D_X, D_Y, embedding_X, embedding_Y = load_checkpoint(opts)
    else:
        G_XtoY, G_YtoX, D_X, D_Y, embedding_X, embedding_Y = create_model(vocab_X, vocab_Y, opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    # iter_X = iter(batch_iterator_X)
    # iter_Y = iter(batch_iterator_Y)

    # test_iter_X = iter(test_batch_iterator_X)
    # test_iter_Y = iter(test_batch_iterator_Y)

    # Get some fixed data from domains X and Y for sampling. These are sentences that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_embedded_sentences_X, fixed_mask_X = get_next_batch_mask(dev_batch_iterator_X, embedding_X)
    fixed_embedded_sentences_X, fixed_mask_X = utils.to_var(fixed_embedded_sentences_X), utils.to_var(
        fixed_mask_X).long()

    fixed_embedded_sentences_Y, fixed_mask_Y = get_next_batch_mask(dev_batch_iterator_Y, embedding_Y)
    fixed_embedded_sentences_Y, fixed_mask_Y = utils.to_var(fixed_embedded_sentences_Y), utils.to_var(
        fixed_mask_Y).long()

    for iteration in range(1, opts.train_iters + 1):

        embedded_sentences_X, mask_X = get_next_batch_mask(batch_iterator_X, embedding_X)
        embedded_sentences_X, mask_X = utils.to_var(embedded_sentences_X), utils.to_var(mask_X).long()

        embedded_sentences_Y, mask_Y = get_next_batch_mask(batch_iterator_Y, embedding_Y)
        embedded_sentences_Y, mask_Y = utils.to_var(embedded_sentences_Y), utils.to_var(mask_Y).long()

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        # Train with real sentences
        d_optimizer.zero_grad()

        m = mask_X.size()[0]
        n = mask_Y.size()[0]

        # 1. Compute the discriminator losses on real sentences
        D_X_loss = loss_helper(D_X.forward(embedded_sentences_X, mask_X), m)
        D_Y_loss = loss_helper(D_Y.forward(embedded_sentences_Y, mask_Y), n)

        d_real_loss = D_X_loss + D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()

        # Train with fake sentences
        d_optimizer.zero_grad()

        # 2. Generate fake sentences that look like domain X based on real sentences in domain Y
        embedded_sentences_X_hat, mask_X_hat, _, _, _ = G_YtoX.forward(embedded_sentences_Y, mask_Y).values()

        # 3. Compute the loss for D_X
        D_X_loss = loss_helper1(D_X.forward(embedded_sentences_X_hat, mask_X_hat), n)

        # 4. Generate fake sentences that look like domain Y based on real sentences in domain X
        embedded_sentences_Y_hat, mask_Y_hat, _, _, _ = G_XtoY.forward(embedded_sentences_X, mask_X).values()

        # 5. Compute the loss for D_Y
        D_Y_loss = loss_helper1(D_Y.forward(embedded_sentences_Y_hat, mask_Y_hat), m)

        d_fake_loss = D_X_loss + D_Y_loss
        d_fake_loss.backward()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        #########################################
        #              Y--X-->Y CYCLE
        #########################################
        g_optimizer.zero_grad()

        # 1. Generate fake sentences that look like domain X based on real sentences in domain Y
        embedded_sentences_X_hat, mask_X_hat, _, _, _ = G_YtoX.forward(embedded_sentences_Y, mask_Y).values()

        # 2. Compute the generator loss based on domain X
        g_loss = loss_helper(D_X.forward(embedded_sentences_X_hat, mask_X_hat), n)

        if opts.no_cycle_consistency_loss == False:
            embedded_sentences_Y_reconstructed, mask_Y_reconstructed, _, _, _ = G_XtoY(embedded_sentences_X_hat,
                                                                                       mask_X_hat, False).values()
            embedded_sentences_Y_reconstructed, mask_Y_reconstructed = utils.trim_tensors(embedded_sentences_Y,
                                                                                          embedded_sentences_Y_reconstructed,
                                                                                          mask_Y_reconstructed)

            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = loss_helper1(embedded_sentences_Y - embedded_sentences_Y_reconstructed,
                                                  n)  # Consider masks?
            # TODO: USE CROSS ENTROPY HERE

            g_loss += cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()

        #########################################
        #             X--Y-->X CYCLE
        #########################################

        g_optimizer.zero_grad()

        # 1. Generate fake sentences that look like domain Y based on real sentences in domain X
        embedded_sentences_Y_hat, mask_Y_hat, _, _, _ = G_XtoY.forward(embedded_sentences_X, mask_X).values()

        # 2. Compute the generator loss based on domain Y
        g_loss = loss_helper(D_Y.forward(embedded_sentences_Y_hat, mask_Y_hat), n)

        if opts.no_cycle_consistency_loss == False:
            embedded_sentences_X_reconstructed, mask_X_reconstructed, _, _, _ = G_YtoX(embedded_sentences_Y_hat,
                                                                                       mask_Y_hat, False).values()
            embedded_sentences_X_reconstructed, mask_X_reconstructed = utils.trim_tensors(embedded_sentences_X,
                                                                                          embedded_sentences_X_reconstructed,
                                                                                          mask_X_reconstructed)

            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = loss_helper1(embedded_sentences_X - embedded_sentences_X_reconstructed, m)
            # TODO: USE CROSS ENTROPY WITH LOGITS FROM RECONSTRUCTED STUFF AND INPUT SENTENCES INDICES
            g_loss += cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                  'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                iteration, opts.train_iters, d_real_loss.data[0], D_Y_loss.data[0],
                D_X_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

        # Save the generated samples
        if iteration % opts.sample_every == 0 or iteration == 1:
            save_samples(iteration, fixed_embedded_sentences_Y, fixed_mask_Y, fixed_embedded_sentences_X, fixed_mask_X,
                         G_YtoX, G_XtoY, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test batch_iterators for sentences from the two domains X and Y
    mono_dataset_reader = MonolingualDatasetReader(lazy=False, max_sent_len=opts.max_sent_len)

    batch_iterator_X, test_batch_iterator_X, vocab_X = get_iterator_vocab("X", mono_dataset_reader, opts)
    batch_iterator_Y, test_batch_iterator_Y, vocab_Y = get_iterator_vocab("Y", mono_dataset_reader, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(batch_iterator_X, batch_iterator_Y,
                  test_batch_iterator_X, test_batch_iterator_Y,
                  vocab_X, vocab_Y, opts)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--no_cycle_consistency_loss', action='store_true', default=False,
                        help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--init_zero_weights', action='store_true', default=False,
                        help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')

    parser.add_argument('--embedding_dim', type=int, default=300, help='The dimention of all embedding layers')

    parser.add_argument('--g_num_layers', type=int, default=1,
                        help="The number of layers for generators's encoder RNN.")
    parser.add_argument('--g_bidirectional', action='store_true', default=True,
                        help='Weather to use bidirectional RNN for generators.')
    parser.add_argument('--g_hidden_size', type=int, default=600,
                        help='The number of hidden units in encoder RNN of generatorss.')
    parser.add_argument('--g_max_decoding_steps', type=int, default=52,
                        help='The number timesteps at decoder of generators. Should be max_sent_len + 2. You should add 2 for <start> and <end> symbols')

    parser.add_argument('--d_hidden_size', type=int, default=600,
                        help='The number of hidden units in encoder RNN of discriminators.')
    parser.add_argument('--d_bidirectional', action='store_true', default=True,
                        help='Weather to use bidirectional RNN for discriminators.')
    parser.add_argument('--d_num_layers', type=int, default=1,
                        help='The number of layers in encoder RNN of discriminators.')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=600,
                        help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of sentences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='The number of threads to use for the batch_iterator.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--path_train_x', type=str, default='europarl/train.et',
                        help='Path to the monolingual train set of lang. X.')
    parser.add_argument('--path_dev_x', type=str, default='europarl/dev.et',
                        help='Path to the monolingual dev set of lang. X.')
    parser.add_argument('--path_train_y', type=str, default='europarl/train.en',
                        help='Path to the monolingual train set of language Y.')
    parser.add_argument('--path_dev_y', type=str, default='europarl/dev.en',
                        help='Path to the monolingual dev set of language Y.')

    parser.add_argument('--Y', type=str, default='Windows', choices=['Apple', 'Windows'],
                        help='Choose the type of sentences for domain Y.')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan_no_cycle')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=800)

    # NLP
    parser.add_argument('--max_vocab_size', type=int, default=None,
                        help='Max size for vocabs. Default to use all tokens')

    parser.add_argument('--max_sent_len', type=int, default=50,
                        help='Removes sentences that are less then max_sent_len tokens long')

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    if opts.no_cycle_consistency_loss == False:
        opts.sample_dir = 'samples_cyclegan'

    if opts.load:
        opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
        opts.sample_every = 20

    print_opts(opts)
    main(opts)
