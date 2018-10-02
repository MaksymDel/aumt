from typing import Dict

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.util import weighted_sum, get_final_encoder_states, sequence_cross_entropy_with_logits
from overrides import overrides
from torch.nn.functional import gumbel_softmax, pad
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

PADDING_SYMBOL = "PADDING_SYMBOL"
# TODO: when reconstructing, consider teacher forcing
#######################################################################################
################################ DISCRIMINATOR NETWORK ################################
#######################################################################################

class Seq2Binary(Model):
    """
    Logistic regression on sentence.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 embedding: Embedding,
                 seq2vec_encoder: Seq2VecEncoder):
        super(Seq2Binary, self).__init__(vocab)

        self._embedding = embedding
        self._encoder = seq2vec_encoder
        self._projection_layer = Linear(self._encoder.get_output_dim(), 1)

    def forward(self,  # type: ignore
                embedded_input: torch.FloatTensor, source_mask: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.
        Parameters
        ----------
        sentence : torch.LongTensor
           Tensor of padded batch of indexed source strings
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        batch_size, _, _ = embedded_input.size()
        final_encoder_output = self._encoder(embedded_input, source_mask)
        logits = self._projection_layer(final_encoder_output)
        probs = torch.sigmoid(logits)

        return probs


#######################################################################################
################################## GENERATOR NETWORK ##################################
#######################################################################################

class VanillaRnn2Rnn(Model):
    """
    Returns predicted indeces
    """

    def __init__(self,
                 source_vocab: Vocabulary,
                 target_vocab: Vocabulary,
                 source_embedding: Embedding,
                 target_embedding: Embedding,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 projection_type: str,
                 gumbel_tau: float,
                 gumbel_hard: bool
                 ) -> None:

        super(VanillaRnn2Rnn, self).__init__(source_vocab)

        self._source_vocab = source_vocab
        self._target_vocab = target_vocab

        self._source_embedder = source_embedding
        self._target_embedder = target_embedding

        self._encoder = encoder

        self._max_decoding_steps = max_decoding_steps

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._target_vocab.get_token_index(START_SYMBOL, "tokens")
        self._end_index = self._target_vocab.get_token_index(END_SYMBOL, "tokens")

        num_classes = self._target_vocab.get_vocab_size("tokens")

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = self._encoder.get_output_dim()

        target_embedding_dim = self._target_embedder.get_output_dim()

        self._decoder_attention = DotProductAttention()
        # The output of attention, a weighted average over encoder outputs, will be
        # concatenated to the input vector of the decoder at each time step.
        self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim

        # TODO: Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

        self._projection_type = projection_type
        self._gumbel_tau = gumbel_tau
        self._gumbel_hard = gumbel_hard

    @overrides
    def forward(self,  # type: ignore
                embedded_input: torch.FloatTensor,
                source_mask: torch.LongTensor,
                targets: torch.LongTensor=None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.
        Parameters
        ----------
        embedded_input : torch.LongTensor
           Embedded input tokens
        source_mask : torch.LongTensor
           Mask that represents batch padding
        targets: torch.LongTensor
            Indexes of the original sequence we are trying to reconstruct.
            If not provided, we do not reconstruct the sequence but try just to generate it
        """
        batch_size, _, _ = embedded_input.size()
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = get_final_encoder_states(encoder_outputs, source_mask,
                                                        self._encoder.is_bidirectional())  # (batch_size, encoder_output_dim)
        if targets is not None:
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        decoder_hidden = final_encoder_output

        decoder_context = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)

        last_predictions = None
        step_logits = []
        step_predictions = []
        step_predictions_softmax = []

        step_embedded_outputs = []
        for timestep in range(num_decoding_steps):
            use_gold_targets = False
            # Use gold tokens when provided during reconstruction
            if targets is not None:
                use_gold_targets = True

            if use_gold_targets:
                input_indices = targets[:, timestep]
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_indices = source_mask.new_full((batch_size,), fill_value=self._start_index)
                else:
                    input_indices = last_predictions  # usually is not differentiable, but I can do it so if I want
            decoder_input = self._prepare_decode_step_input(input_indices, decoder_hidden,
                                                            encoder_outputs, source_mask)
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                 (decoder_hidden, decoder_context))

            # (batch_size, num_classes)
            output_projections = self._output_projection_layer(decoder_hidden)
            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # GUMBEL SOFTMAX
            if self._projection_type == 'gumbel':
                token_types_weights = gumbel_softmax(logits=output_projections, hard=self._gumbel_hard, tau=self._gumbel_tau)
            elif self._projection_type == 'softmax':
                token_types_weights = F.softmax(output_projections, dim=-1)
            elif self._projection_type == 'direct':
                raise NotImplementedError("direct vector prediction is not supportet at the moment")
            else:
                raise NotImplementedError("wrong projection type value; possible are [gumbel | softmax | direct]")

            softmax_classes = torch.argmax(output_projections, dim=-1).long() # sampling after softmax operation
            predicted_classes = torch.argmax(token_types_weights, dim=-1).long() # sampling after custom projection
            # (batch_size, 1)
            # we pass there softmax golden labels always
            last_predictions = predicted_classes  # usually it is not differentiable, but I can do it so if I want

            step_predictions.append(predicted_classes.unsqueeze(1))
            step_predictions_softmax.append(softmax_classes.unsqueeze(1))

            embedded_output_tokens = token_types_weights.matmul(self._target_embedder.weight) # still differentiable
            step_embedded_outputs.append(embedded_output_tokens.unsqueeze(1))

        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        all_predictions = torch.cat(step_predictions, 1)
        all_predictions_softmax = torch.cat(step_predictions_softmax, 1)

        # finilize predictions and compute mask
        predicted_tokens_softmax, predicted_indices_softmax = self._finize_predictions(all_predictions_softmax)
        predicted_tokens, predicted_indices = self._finize_predictions(all_predictions)

        all_embedded_output_tokens = torch.cat(step_embedded_outputs, 1)
        # self._get_mask_from_indices(predicted_indices_softmax)
        mask = self._get_mask_from_indices(predicted_indices)
        maxlen = mask.size()[1]
        # trim embedded output tokens to match the mask
        all_embedded_output_tokens = all_embedded_output_tokens[:, :maxlen, :]

        output_dict = {}

        output_dict["embedded_output_tokens"] = all_embedded_output_tokens
        output_dict["mask"] = mask
        output_dict["predicted_tokens"] = predicted_tokens
        output_dict["predicted_tokens_softmax"] = predicted_tokens_softmax
        output_dict["logits"] = logits

        if targets is not None:  # if there are targets, we compute reconstruction loss
            target_mask = targets != self._target_vocab.get_token_index(PADDING_SYMBOL)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["cycle_loss"] = loss
            # TODO: Define metrics

        return output_dict

    def _prepare_decode_step_input(self,
                                   input_indices: torch.LongTensor,
                                   decoder_hidden_state: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None,
                                   encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.
        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.
        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        input_indices = input_indices.long()
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = self._target_embedder(input_indices)  # this should be sperate func that work with different
        # forms of in

        # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
        # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
        # complain.
        encoder_outputs_mask = encoder_outputs_mask.float()
        # (batch_size, input_sequence_length)
        input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
        # (batch_size, encoder_output_dim)
        attended_input = weighted_sum(encoder_outputs, input_weights)
        # (batch_size, encoder_output_dim + target_embedding_dim)
        return torch.cat((attended_input, embedded_input), -1)

    def _finize_predictions(self, predicted_indices):
        "strips ids till the first end symbol "
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        all_predicted_indices = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                ind_end_symbol = indices.index(self._end_index)
                if ind_end_symbol == 0:  # if empty line is predicted
                    ind_end_symbol = 1  # do not allow for empty lines
                indices = indices[:ind_end_symbol]

            predicted_tokens = [self._target_vocab.get_token_from_index(x, namespace="tokens")
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
            all_predicted_indices.append(indices)

        return all_predicted_tokens, all_predicted_indices

    def _get_mask_from_indices(self, all_predicted_indices):
        lens = [len(l) for l in all_predicted_indices]
        maxlen = max(lens)
        mask = numpy.arange(maxlen) < numpy.array(lens)[:, None]  # key line
        mask = torch.from_numpy(mask.astype(int)).long()

        if torch.cuda.is_available():  # optionaly move mask to GPU
            mask = mask.cuda()
        return mask

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss