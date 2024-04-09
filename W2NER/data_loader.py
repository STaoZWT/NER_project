import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import numpy as np
# import prettytable as pt
# from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"
use_author_ner_format = True

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

TAGS = ['O',
 'B-PERSON',
 'I-PERSON',
 'B-NORP',
 'I-NORP',
 'B-FAC',
 'I-FAC',
 'B-ORG',
 'I-ORG',
 'B-GPE',
 'I-GPE',
 'B-LOC',
 'I-LOC',
 'B-PRODUCT',
 'I-PRODUCT',
 'B-DATE',
 'I-DATE',
 'B-TIME',
 'I-TIME',
 'B-PERCENT',
 'I-PERCENT',
 'B-MONEY',
 'I-MONEY',
 'B-QUANTITY',
 'I-QUANTITY',
 'B-ORDINAL',
 'I-ORDINAL',
 'B-CARDINAL',
 'I-CARDINAL',
 'B-EVENT',
 'I-EVENT',
 'B-WORK_OF_ART',
 'I-WORK_OF_ART',
 'B-LAW',
 'I-LAW',
 'B-LANGUAGE',
 'I-LANGUAGE']

# basically for label <-> id mapping
class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        # print("instance is ", instance)
        """
        if len(instance['sentence']) == 0:
            continue
        """

        # use tokenizer to turn word into tokens
        tokenized_input = [tokenizer.tokenize(word) for word in instance['tokens']]
        """ for example
        word_array = ["Helloaaa", ",", "this", "is", "one", "sentence", "split", "into", "words", "."]
        tokens = [['ĠHello', 'aaa'],
                  ['Ġ,'],
                  ['Ġthis'],
                  ['Ġis'],
                  ['Ġone'],
                  ['Ġsentence'],
                  ['Ġsplit'],
                  ['Ġinto'],
                  ['Ġwords'],
                  ['Ġ.']]
        """

        # collect each word pieces -> a list of pieces
        pieces = [piece for pieces in tokenized_input for piece in pieces]
        """ pieces = ['ĠHello',
          'aaa',
          'Ġ,',
          'Ġthis',
          'Ġis',
          'Ġone',
          'Ġsentence',
          'Ġsplit',
          'Ġinto',
          'Ġwords',
          'Ġ.']
        """
        # use the vocab from tokenizer to convert pieces to ids
        # _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = tokenizer(instance["tokens"], is_split_into_words=True)
        _bert_inputs = _bert_inputs["input_ids"]

        # [CLS] + input ids + [SEP]
        # _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        # num of words in a sentence
        length = len(instance['tokens'])

        _grid_labels = np.zeros((length, length), dtype=int)

        # [num_word x num_bert_input]
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)

        # [num_word x num_word]
        _dist_inputs = np.zeros((length, length), dtype=int)

        # [num_word x num_word]
        _grid_mask2d = np.ones((length, length), dtype=bool)


        # for each word in a sentence, create a piece to word mapping, (to find corresponding word later? word -> tokenized pieces -> word)
        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokenized_input):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                # index -> pieces
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)
        
        # don't know what _dist_inputs does
        # but for the example above, output is like
        """
        array([[ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9],
              [ 1,  0, -1, -2, -3, -4, -5, -6, -7, -8],
              [ 2,  1,  0, -1, -2, -3, -4, -5, -6, -7],
              [ 3,  2,  1,  0, -1, -2, -3, -4, -5, -6],
              [ 4,  3,  2,  1,  0, -1, -2, -3, -4, -5],
              [ 5,  4,  3,  2,  1,  0, -1, -2, -3, -4],
              [ 6,  5,  4,  3,  2,  1,  0, -1, -2, -3],
              [ 7,  6,  5,  4,  3,  2,  1,  0, -1, -2],
              [ 8,  7,  6,  5,  4,  3,  2,  1,  0, -1],
              [ 9,  8,  7,  6,  5,  4,  3,  2,  1,  0]])

        turn to 

        array([[19, 10, 11, 11, 12, 12, 12, 12, 13, 13],
              [ 1, 19, 10, 11, 11, 12, 12, 12, 12, 13],
              [ 2,  1, 19, 10, 11, 11, 12, 12, 12, 12],
              [ 2,  2,  1, 19, 10, 11, 11, 12, 12, 12],
              [ 3,  2,  2,  1, 19, 10, 11, 11, 12, 12],
              [ 3,  3,  2,  2,  1, 19, 10, 11, 11, 12],
              [ 3,  3,  3,  2,  2,  1, 19, 10, 11, 11],
              [ 3,  3,  3,  3,  2,  2,  1, 19, 10, 11],
              [ 4,  3,  3,  3,  3,  2,  2,  1, 19, 10],
              [ 4,  4,  3,  3,  3,  3,  2,  2,  1, 19]])
        """
        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        # 这里 每个entity 可能由一个或多个word 组成，比如 人名："Dick", "Spring"-> index: [24,25], label "PER"
        # 但是，我处理的数据，大概长这样子: 
        # [Many, overseas, Chinese, and, community, groups, gathered, there, to, wait, for, the, commencement, of, the, Chinese, Culture, and, Costume, Parade, .]
        # [O, O, B-NORP, O, O, O, O, O, O, O, O, O, O, O, B-EVENT, I-EVENT, I-EVENT, I-EVENT, I-EVENT, I-EVENT, O]
        # W2NER 想要的可能是, index: [14,15,16,17,18,19], label: EVENT
        # 而我这里，label 有一些差异
        # 顺便，这里提到了:
        # 作者没有针对文章提出的标记方案和BIO、Span等方案做过对比
        # 我们用的标记和作者的还不一样
        # https://github.com/ljynlp/W2NER/issues/13
        # 代码中，影响主要是_grid_labels 这个参数，应该还影响precision,recall
        
        _entity_text = None

        if use_author_ner_format:
            for entity in instance["ner"]:
                # print("use au ner for, entity", entity)
                index = entity["index"]
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_labels[index[i], index[i + 1]] = 1
                _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

            _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])
        else:
            for oldindex, ner_tag in enumerate(instance["ner_tags"]):
                # index = entity["index"]
                # print(oldindex, ner_tag)
                if ner_tag == "O":
                    continue
                index = [oldindex]
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_labels[index[i], index[i + 1]] = 1
                _grid_labels[index[-1], index[0]] = vocab.label_to_id(ner_tag)

            _entity_text = set([utils.convert_index_to_text([index], vocab.label_to_id(tag)) for index, tag in enumerate(instance["ner_tags"]) if tag != "O"  ])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


"""
  {
    "sentence": [
      "Charlton",
      ",",
      "61",
      ",",
      "and",
      "his",
      "wife",
      ",",
      "Peggy",
      ",",
      "became",
      "citizens",
      "of",
      "Ireland",
      "when",
      "they",
      "formally",
      "received",
      "Irish",
      "passports",
      "from",
      "deputy",
      "Prime",
      "Minister",
      "Dick",
      "Spring",
      "who",
      "said",
      "the",
      "honour",
      "had",
      "been",
      "made",
      "in",
      "recognition",
      "of",
      "Charlton",
      "'s",
      "achievements",
      "as",
      "the",
      "national",
      "soccer",
      "manager",
      "."
    ],
    "ner": [
      {
        "index": [
          0
        ],
        "type": "PER"
      },
      {
        "index": [
          8
        ],
        "type": "PER"
      },
      {
        "index": [
          13
        ],
        "type": "LOC"
      },
      {
        "index": [
          18
        ],
        "type": "MISC"
      },
      {
        "index": [
          24,
          25
        ],
        "type": "PER"
      },
      {
        "index": [
          36
        ],
        "type": "PER"
      }
    ]
  }
"""

def fill_vocab(vocab, dataset):
    """
    entity_num = 0
    for instance in dataset:
        # put all the ner tags into vocab
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        # count the number of entities in a sentence and -> entity_num
        entity_num += len(instance["ner"])
    return entity_num
    """
    pass

def convert_format(example):
    # tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    ner_json = []
    prev_dict = {}
    prev_label_tail = None
    prev_index = []
    new_ner_tags = example["ner_tags"][:]
    new_ner_tags.append("O")
    for i, label in enumerate(new_ner_tags):
        new_label_tail = label.split("-")[-1]
        new_label_head = label.split("-")[0]
        if i > 0 and prev_label_tail and new_label_tail != prev_label_tail and (new_label_head == "B" or new_label_head == "O"):
            prev_dict["index"] = prev_index
            prev_dict["type"] = prev_label_tail
            ner_json.append(json.loads(json.dumps(prev_dict)))
            prev_dict = {}
            prev_label_tail = None
            prev_index = []
        elif label != "O":
            prev_label_tail = new_label_tail
            prev_index.append(i)

    example["ner"] = ner_json
    return example

def process_tags(examples):
    new_tags = []
    for tag in examples:
        new_tag = tag.split("-")[-1]
        new_tags.append(new_tag)
    return new_tags

def load_data_bert(config):
    datasets = load_dataset('./data/englishv12')
   
    print("暂时跑10000个sample, 哎")
    #train_data = load_dataset('./data/englishv12',split='train[:10000]')
    #dev_data = load_dataset('./data/englishv12',split='validation[:1000]')
    #test_data = load_dataset('./data/englishv12',split='test[:1000]')
    train_data = datasets["train"]
    dev_data = datasets["validation"]
    test_data = datasets["test"]
    
    # update_datasets = datasets.map(convert_format, batched=False, load_from_cache_file=False)
    update_datasets = datasets.map(convert_format, batched=False)
    # updated_train_data = load_dataset('./data/englishv12',split='train[:10000]')
    # updated_dev_data = load_dataset('./data/englishv12',split='validation[:1000]')
    # updated_test_data = load_dataset('./data/englishv12',split='test[:1000]')
    # updated_train_data = updated_train_data.map(convert_format, batched=False, load_from_cache_file=False)
    # updated_dev_data = updated_dev_data.map(convert_format, batched=False, load_from_cache_file=False)
    # updated_test_data = updated_test_data.map(convert_format, batched=False, load_from_cache_file=False)
    
    updated_train_data = update_datasets["train"]
    updated_dev_data = update_datasets["validation"]
    updated_test_data = update_datasets["test"]

    
    # obtain the tokenizer, add_prefix_space=True for roberta
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/", add_prefix_space=True)

    vocab = Vocabulary()
    
    label_list = datasets["train"].features["ner_tags_index"].feature.names
    updated_label_list = process_tags(datasets["train"].features["ner_tags_index"].feature.names)

    # create label <-> id mapping
    if use_author_ner_format:
        for label in updated_label_list:
            print(label)
            vocab.add_label(label)
    else:
        for label in label_list:
            print(label)
            vocab.add_label(label)

    # number of ner tags in sentences + <unk> + <pad> + <suc>
    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = None
    dev_dataset = None
    test_dataset = None
    ori_train_data = None
    ori_dev_data = None
    ori_test_data = None

    if use_author_ner_format:
        train_dataset = RelationDataset(*process_bert(updated_train_data, tokenizer, vocab))
        dev_dataset = RelationDataset(*process_bert(updated_dev_data, tokenizer, vocab))
        test_dataset = RelationDataset(*process_bert(updated_test_data, tokenizer, vocab))
        ori_train_data = updated_train_data
        ori_dev_data = updated_dev_data
        ori_test_data = updated_test_data
    else:
        train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
        dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
        test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
        ori_train_data = train_data
        ori_dev_data = dev_data
        ori_test_data = test_data
    return (train_dataset, dev_dataset, test_dataset), (ori_train_data, ori_dev_data, ori_test_data)

