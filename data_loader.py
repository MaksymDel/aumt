from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
# from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides


def get_iterator_vocab(lang, mono_dataset_reader, opts):
    if lang == 'X':
        path_train = opts.path_train_x
        path_dev = opts.path_dev_x
    elif lang == 'Y':
        path_train = opts.path_train_y
        path_dev = opts.path_dev_y

    instances_train = mono_dataset_reader.read(path_train)
    instances_dev = mono_dataset_reader.read(path_dev)

    vocab = Vocabulary.from_instances(instances=instances_train + instances_dev, max_vocab_size=opts.max_vocab_size)

    # iterator_creator = BucketIterator(sorting_keys = [("sentence", "num_tokens")], batch_size=32, max_instances_in_memory=None)
    iterator_creator = BasicIterator(batch_size=opts.batch_size, max_instances_in_memory=None)
    iterator_creator.index_with(vocab)

    batch_iterator_train = iterator_creator(instances=instances_train, num_epochs=None, shuffle=False)
    batch_iterator_dev = iterator_creator(instances=instances_dev, num_epochs=None, shuffle=False)

    return batch_iterator_train, batch_iterator_dev, vocab


class MonolingualDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False, max_sent_len=50) -> None:
        super().__init__(lazy)
        self._sentence_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._sentence_token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._sentence_add_start_token = True
        self._max_sent_len = max_sent_len

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            print("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line = line.lower()
                tokenized_sentence = self._sentence_tokenizer.tokenize(line)
                if len(tokenized_sentence) > self._max_sent_len:
                    continue

                yield self.text_to_instance(tokenized_sentence)

    @overrides
    def text_to_instance(self, tokenized_sentence) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        if self._sentence_add_start_token:
            tokenized_sentence.insert(0, Token(START_SYMBOL))
        tokenized_sentence.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)

        return Instance({'sentence': sentence_field})
