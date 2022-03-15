#!/usr/bin/env python
# coding: utf-8

# # Benchmarking Framework on ASAP-AES Source-Dependent Prompts
# The purpose of this notebook is to evaluate our hybrid framework on ASAP-AES prompts 3, 4, 5 and 6. We will test combinations of:
# 1. A + B = BERT + Essay-level features
# 2. A + B + C = BERT + Essay-level features + Keyword attention features
# 
# We will also test with both the full dataset and 50% of the dataset.
'''
We employ the same KFold 60%-20%-20% split as Taghipour & Ng (2016).
'''

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import implementations
import logging
from .model_arguments import ModelArguments, MultimodalDataTrainingArguments

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers import TrainingArguments
# from transformers.training_args import TrainingArguments

from implementations.data import load_data_from_folder
from implementations.model import TabularConfig
from implementations.model import AutoModelWithTabular
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
import json

import argparse

logging.basicConfig(level=logging.INFO)
os.environ['COMET_MODE'] = 'DISABLED'

import warnings
warnings.filterwarnings("ignore")


# ### Reading Data
# We choose Set 3, Set 4, Set 5 and Set 6. These are source-dependent scoring prompts, where students read an excerpt before answering questions.

# In[2]:

set3 = pd.read_csv('./asap_aes/set3_features.csv')
set4 = pd.read_csv('./asap_aes/set4_features.csv')
set5 = pd.read_csv('./asap_aes/set5_features.csv')
set6 = pd.read_csv('./asap_aes/set6_features.csv')


# In[3]:

num_cols = ['num_words', 'num_sentences', 'num_lemmas',
       'num_commas', 'num_exclamation_marks', 'num_question_marks',
       'average_word_length', 'average_sentence_length', 'num_nouns',
       'num_verbs', 'num_adjectives', 'num_adverbs', 'num_conjunctions',
       'num_spelling_errors', 'num_stopwords', 'automated_readability_index',
       'coleman_liau_index', 'dale_chall_index', 'difficult_word_count',
       'flesch_kincaid_grade', 'gunning_fog', 'linsear_write_formula',
       'smog_index', 'syllables_count']
cat_cols = ['temp_cat']
text_cols = ['essay']
label_col = 'domain1_score'

column_info_dict = {
    'text_cols': text_cols,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'label_col': label_col,
    'label_list': [0, 1, 2, 3, 4]
}

model_args = ModelArguments(
    model_name_or_path='bert-base-uncased'
)

data_args = MultimodalDataTrainingArguments(
    data_path='./asap_aes',
    combine_feat_method='mlp_on_concatenated_cat_and_numerical_feats_then_concat',
    column_info=column_info_dict,
    task='classification',
    use_simple_classifier=True
)

tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
print('Specified tokenizer: ', tokenizer_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path_or_name,
    cache_dir=model_args.cache_dir,
    max_sequence_length=150
)

import numpy as np
from scipy.special import softmax
import shutil

from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
    accuracy_score
)

GLOBAL_PREDICTIONS = None

def calc_classification_metrics(p: EvalPrediction):
  pred_labels = np.argmax(p.predictions[0], axis=1)
  pred_scores = softmax(p.predictions[0], axis=1)
  labels = p.label_ids
  acc = (pred_labels == labels).mean()
  f1 = f1_score(y_true=labels, y_pred=pred_labels, average='weighted')
  result = {
      "acc": acc,
      "f1": f1,
      "acc_and_f1": (acc + f1) / 2,
      "mcc": matthews_corrcoef(labels, pred_labels),
      "QWK": cohen_kappa_score(labels, pred_labels, weights='quadratic')
  }
  return result

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

from sklearn.model_selection import train_test_split

def remove_percentage_of_data(essay_set, percentage):
    train, test = train_test_split(essay_set, test_size=percentage, random_state=42, stratify=essay_set['domain1_score'])
    return train

def make_factor_of_df(essay_set, factor=32):
    if len(essay_set) % factor != 0:
        num_to_add = factor - len(essay_set) % factor
        essay_set = essay_set.append(essay_set[:num_to_add])
    return essay_set

def make_factor_of(essay_set, factor=32):
    if len(essay_set) % factor != 0:
        num_to_add = factor - len(essay_set) % factor
        for i in range(num_to_add):
            essay_set.append(essay_set[0 + i])
    print('Length of set: ', len(essay_set))
    print('Type of set: ', type(essay_set))
    return essay_set

def find_best_fold(eval_results):
    qwks = {}
    for fold in eval_results:
        qwks[fold] = eval_results[fold]['eval_QWK']

    return max(zip(qwks.values(), qwks.keys()))[1]

def train_model_kfold(set_num, percentage_of_training=1.0, label_list=[0, 1, 2, 3]):
    print(f"TRAINING SET: {set_num}")

    all_data = pd.read_csv('/home/mmu-user/benchmarking/asap_aes_kfold/training_set_rel3_features.tsv', sep='\t', encoding='ISO-8859-1', index_col=0)

    fold_count = 1
    glove_tokenizer = Tokenizer(num_words=10000)
    glove_tokenizer.fit_on_texts(all_data[all_data['essay_set'] == set_num]['essay'].to_list())
    evaluation = {}
    models = {}
    num_words = int(all_data[all_data['essay_set'] == set_num]['num_words'].max())

    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    print('Specified tokenizer: ', tokenizer_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=model_args.cache_dir,
        max_sequence_length=num_words
    )
    
    for fold in range(5):
        print(f'Fold {fold_count}')

        train = pd.read_csv(f'./asap_aes_kfold/fold_{fold}/train.tsv', sep='\t', encoding='ISO-8859-1')
        val = pd.read_csv(f'./asap_aes_kfold/fold_{fold}/dev.tsv', sep='\t', encoding='ISO-8859-1')
        test = pd.read_csv(f'./asap_aes_kfold/fold_{fold}/test.tsv', sep='\t', encoding='ISO-8859-1')

        train = train[train['essay_set'] == set_num]
        val = val[val['essay_set'] == set_num]
        test = test[test['essay_set'] == set_num]

        train['temp_cat'] = 1
        val['temp_cat'] = 1
        test['temp_cat'] = 1
        # setting lemmatized to empty string to avoid errors when no keywords are used
        train['lemmatized'] = ' '
        val['lemmatized'] = ' '
        test['lemmatized'] = ' '

        if percentage_of_training < 1.0:
            train = remove_percentage_of_data(train, percentage_of_training)

        train = make_factor_of_df(train)
        val = make_factor_of_df(val)
        test = make_factor_of_df(test)

        train.to_csv(f'./asap_aes/train.csv', index=False)
        val.to_csv(f'./asap_aes/val.csv', index=False)
        test.to_csv(f'./asap_aes/test.csv', index=False)

        train_dataset, val_dataset, test_dataset = load_data_from_folder(
            f'./asap_aes',
            data_args.column_info['text_cols'],
            tokenizer,
            label_col=data_args.column_info['label_col'],
            label_list=data_args.column_info['label_list'],
            categorical_cols=data_args.column_info['cat_cols'],
            numerical_cols=data_args.column_info['num_cols'],
            sep_text_token_str=tokenizer.sep_token,
            max_token_length=num_words,
            glove_tokenizer=glove_tokenizer,
            keywords=['keyword1', 'keyword2', 'keyword3'],
            max_keyword_length=10,
        )

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=f'./cache/set_{set_num}/fold_{fold_count}',
        )

        tabular_config = TabularConfig(num_labels=5,
                                        cat_feat_dim=train_dataset.cat_feats.shape[1],
                                        numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                        keyword_attention_dim=0,
                                        vocab_size=0,
                                        num_keywords = len(['keyword1', 'keyword2', 'keyword3']),
                                        keyword_MLP_out_dim=0,
                                        save_attentions=False,
                                        attentions_path='./asap_aes/attentions/',
                                        add_attention_module=False,
                                        batch_size=32,
                                        num_words=num_words,
                                    **vars(data_args))
                                    
        config.tabular_config = tabular_config

        model = AutoModelWithTabular.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            config=config,
            cache_dir=f'./cache/set_{set_num}/fold_{fold_count}',
        )

        training_args = TrainingArguments(
            output_dir = f'./asap_aes_kfold/output/set_{set_num}_{percentage_of_training}/fold_{fold_count}',
            num_train_epochs = 4,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,    
            per_device_eval_batch_size=16,
            eval_accumulation_steps=2,
            evaluation_strategy = "epoch",
            save_total_limit = 1,
            disable_tqdm = False,
            load_best_model_at_end=True,
            logging_steps=1,
            run_name = 'longformer-B'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=calc_classification_metrics,
        )

        print(f'Training model on fold {fold_count}')
        trainer.train()

        print(f'Validating model on fold {fold_count}')
        eval_data = trainer.evaluate()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=calc_classification_metrics,
        )

        print(f'Testing model on fold {fold_count}')
        test_data = trainer.evaluate()
    
        print(test_data)

        evaluation[f'fold_{fold_count}'] = test_data
        # models[f'fold_{fold_count}'] = model

        remove(f'./cache/set_{set_num}/fold_{fold_count}')        

        fold_count += 1

        torch.cuda.empty_cache()
    
    # only keep best fold
    best_fold = find_best_fold(evaluation)
    print(f'Best fold is {best_fold}')
    to_remove = [fold for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'] if fold != best_fold]
    for i in to_remove:
        remove(f'./asap_aes_kfold/output/set_{set_num}_{percentage_of_training}/{i}')

    return models, evaluation


keywords = {
    3: ['trying to keep my balance in my dehydrated state',
             'no one in sight, not a building, car, or structure of any kind',
             'flat road was replaced by short rolling hills',
             'hitting my water bottles pretty regularly',
             'somewhere in the neighborhood of two hundred degrees',
             'sun was beginning to beat down',
             'drop from heatstroke on a gorgeous day',
             'water bottles contained only a few tantalizing sips',
             'high deserts of California',
             'enjoyed the serenity of an early-summer evening',
             'traveling through the high deserts of California in June',
             'ROUGH ROAD AHEAD: DO NOT EXCEED POSTED SPEED LIMIT',
             'long, crippling hill',
             'tarlike substance followed by brackish water',
             'thriving little spot at one time',
             'fit the traditional definition of a ghost town',
             'Wide rings of dried sweat circled my shirt',
             'growing realization that I could drop from heatstroke on a gorgous day in June',
             'water bottle contained only a few tantalizing sips',
             'wide rings of dried sweat circled my shirt',
             'checked my water supply',
             'brackish water faling somewhere in the neighborhood of two hundred degrees',
             'birds would pick me clean'],
    4: ['will take that test again', 'many of the things that she had thought of as strange', 'not like the kind we had before', 'I failed the test', 'rich sweet scent', 'when the snow melts', 'geese return', 'hibiscus is budding', 'gentle grandmother', 'distinctive V was etched against the evening sky', 'familiar fragrance filled her lungs', 'could almost feel light strands of her grandmother long gray hair', 'attitude towards her new country and her driving test', 'hibiscus plant in the winter is not as beautiful in the bitter cold', 'adapts and survives', 'returns to its beautiful state in the spring', 'bitter about her new country and driving test', 'new start or new opportunity', 'memories of home', 'overcoming her obstacles', 'noticed tear stains on her a daughter cheeks and her puffy eyes', 'symbolize change and adoption', 'make it through the winter into the spring', 'life is blooming', 'she was still too shaky to say the words at home', 'bitter melon'],
    5: ['always be grateful to my parents for their love and sacrifice', 'rich culinary skills', 'love of cooking', 'passionate Cuban music', 'aromas of the kitchen', 'innocence of childhood', 'congregation of family and friends', 'endless celebrations', 'warm home', 'came together in great solidarity and friendship', 'close-knit community of honest, hardworking immigrants', 'kept their arms and door open to the many people we considered family', 'came selflessly', 'struggled both personally and financially', 'facing cultural hardships', 'overt racism was the norm', 'drove them to endure these hard times', 'strength and perseverance', 'love and sacrifice', 'spirit of generosity impressed upon me at such an early age', 'demonstration of how important family and friends are', 'teachings have been basis of my life', 'warmth of the kitchen', 'humble house', 'not just scent and music but life and love', 'definition of family', 'rich culinary skills', 'never forget how my parents turned this simple house into a home'],
    6: ['one of safety', 'dirigibles from outside of the United States used hydrogen instead of helium', 'nature itself', 'winds on top of the building were constantly shifting', 'violent air currents', 'law against airships flying too low over urban areas', 'moored in open landing fields', 'could be weighted down in the back with lead weights', 'dangling high above pedestrians on the street was neither practical nor safe', 'swivel around and around the mooring mast', 'how much worse that accident could have been', 'could not simply drop a mooring mast on top of the empire state building flat roof', 'stress of the dirigible load', 'mooring air ships to a fixed mast', 'puncture the dirigible shell', 'neither practical nor safe']
}

def get_keywords(set_name):
    return keywords[set_name]


import torch
from torchtext.legacy.data import Field
import torchtext

def load_embeddings(data):
    text_field = Field(
        sequential=True,
        tokenize='basic_english', 
        # fix_length=5,
        lower=True
    )

    label_field = Field(sequential=False, use_vocab=False)

    # sadly have to apply preprocess manually
    preprocessed_text = data.apply(
        lambda x: text_field.preprocess(x)
    )

    # load fastext simple embedding with 300d
    text_field.build_vocab(
        preprocessed_text, 
        vectors=torchtext.vocab.Vectors("../glove.6B.300d.txt"),
        max_size=10000,
        vectors_cache='./no-cache-2/'
    )
    # get the vocab instance
    vocab = text_field.vocab

    return vocab

def clean_oov(data, vocab_size):
    train_dataset_clean = []
    for i in data:
        if (not [i for i, x in enumerate((i['lemmatized_answer_tokens'] >= vocab_size)) if x]):
            train_dataset_clean.append(i)
    print('Before clean size: ', len(data))
    print('After clean size: ', len(train_dataset_clean))
    return train_dataset_clean

def train_model_with_keywords_kfold(set_num, percentage_of_training=1.0, freeze_bert=False, bert_base_path='', epochs=4, label_list=[0, 1, 2, 3]):
    print(f"TRAINING SET: {set_num}")

    all_data = pd.read_csv('/home/mmu-user/benchmarking/asap_aes_kfold/training_set_rel3_features.tsv', sep='\t', encoding='ISO-8859-1', index_col=0)
    all_data['essay'] = all_data['essay'].apply(lambda x: x.replace(r'/[^\w,.:;\[\]()/\!@#$%^&*+{}<>=?~|" -]/g', ''))
    all_data['essay'] = all_data['essay'].apply(lambda x: x.replace(r'/\s+/g', ''))

    num_words = int(all_data[all_data['essay_set'] == set_num]['num_words'].max())

    keyword_list = get_keywords(set_num)

    fold_count = 1
    glove_tokenizer = Tokenizer(num_words=10000)
    glove_tokenizer.fit_on_texts(all_data[all_data['essay_set'] == set_num]['essay'].to_list() + keyword_list)
    evaluation = {}
    models = {}
    num_words = int(all_data[all_data['essay_set'] == set_num]['num_words'].max())

    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    print('Specified tokenizer: ', tokenizer_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=model_args.cache_dir,
        max_sequence_length=num_words
    )
    
    for fold in range(5):
        print(f'Fold {fold_count}')

        train = pd.read_csv(f'./asap_aes_kfold/fold_{fold}/train.tsv', sep='\t', encoding='ISO-8859-1')
        val = pd.read_csv(f'./asap_aes_kfold/fold_{fold}/dev.tsv', sep='\t', encoding='ISO-8859-1')
        test = pd.read_csv(f'./asap_aes_kfold/fold_{fold}/test.tsv', sep='\t', encoding='ISO-8859-1')

        train['essay'] = train['essay'].apply(lambda x: x.replace(r'/[^\w,.:;\[\]()/\!@#$%^&*+{}<>=?~|" -]/g', ''))
        train['essay'] = train['essay'].apply(lambda x: x.replace(r'/\s+/g', ''))
        val['essay'] = val['essay'].apply(lambda x: x.replace(r'/[^\w,.:;\[\]()/\!@#$%^&*+{}<>=?~|" -]/g', ''))
        val['essay'] = val['essay'].apply(lambda x: x.replace(r'/\s+/g', ''))
        test['essay'] = test['essay'].apply(lambda x: x.replace(r'/[^\w,.:;\[\]()/\!@#$%^&*+{}<>=?~|" -]/g', ''))
        test['essay'] = test['essay'].apply(lambda x: x.replace(r'/\s+/g', ''))

        train = train[train['essay_set'] == set_num]
        val = val[val['essay_set'] == set_num]
        test = test[test['essay_set'] == set_num]

        train['temp_cat'] = 1
        val['temp_cat'] = 1
        test['temp_cat'] = 1

        train['lemmatized'] = train['essay']
        val['lemmatized'] = val['essay']
        test['lemmatized'] = test['essay']

        if percentage_of_training < 1.0:
            train = remove_percentage_of_data(train, percentage_of_training)

        train.to_csv(f'./asap_aes/train.csv', index=False)
        val.to_csv(f'./asap_aes/val.csv', index=False)
        test.to_csv(f'./asap_aes/test.csv', index=False)

        print('Labels: ', label_list)

        column_info_dict = {
            'text_cols': text_cols,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'label_col': label_col,
            'label_list': label_list
        }

        data_args = MultimodalDataTrainingArguments(
            data_path='./asap_aes',
            combine_feat_method='mlp_on_concatenated_cat_and_numerical_feats_then_concat',
            column_info=column_info_dict,
            task='classification',
            use_simple_classifier=True
        )

        train_dataset, val_dataset, test_dataset = load_data_from_folder(
            f'./asap_aes',
            data_args.column_info['text_cols'],
            tokenizer,
            label_col=data_args.column_info['label_col'],
            label_list=data_args.column_info['label_list'],
            categorical_cols=data_args.column_info['cat_cols'],
            numerical_cols=data_args.column_info['num_cols'],
            sep_text_token_str=tokenizer.sep_token,
            max_token_length=num_words,
            glove_tokenizer=glove_tokenizer,
            keywords=keyword_list,
            max_keyword_length=max([len(i.split()) for i in keyword_list]),
        )

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=f'./cache/set_{set_num}',
        )

        vocab = load_embeddings(pd.Series(all_data[all_data['essay_set'] == set_num]['essay'].tolist() + keyword_list))    

        print(f'tokenizer vocab size: {len(glove_tokenizer.word_index)}')
        print(f'glove vocab size: {len(vocab)}')

        tabular_config = TabularConfig(num_labels=5,
                                        cat_feat_dim=train_dataset.cat_feats.shape[1],
                                        numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                        vocab_size=len(vocab),
                                        num_keywords = len(keyword_list),
                                        keyword_attention_dim=100,
                                        keyword_MLP_out_dim=100,
                                        save_attentions=False,
                                        attentions_path='./asap_aes/attentions/',
                                        add_attention_module=True,
                                        batch_size=16,
                                        max_keyword_length=max([len(i.split()) for i in keyword_list]),
                                        num_words=num_words,
                                    **vars(data_args))
        config.tabular_config = tabular_config

        train_dataset = clean_oov(train_dataset, len(glove_tokenizer.word_index) - 2)
        val_dataset = clean_oov(val_dataset, len(glove_tokenizer.word_index) - 2)
        test_dataset = clean_oov(test_dataset, len(glove_tokenizer.word_index) - 2)

        train_dataset = make_factor_of(train_dataset)
        val_dataset = make_factor_of(val_dataset)
        test_dataset = make_factor_of(test_dataset)

        model = AutoModelWithTabular.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                config=config,
                cache_dir=f'./cache/set_{set_num}',
        )

        if freeze_bert:
            base_dict = torch.load(bert_base_path)
            bert_dict = {k: v for k, v in base_dict.items() if k in model.state_dict() and 'bert' in k}
            model.load_state_dict(bert_dict, strict=False)
            for name, param in model.named_parameters():
                if 'bert' in name:
                    param.requires_grad = False

        pretrained_embeddings = vocab.vectors
        model.embedding_layer.weight.data = pretrained_embeddings.cuda()    

        training_args = TrainingArguments(
                output_dir = f'./asap_aes_kfold/output/set_{set_num}_{percentage_of_training}_keyword/fold_{fold_count}',
                num_train_epochs = epochs,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=2,    
                per_device_eval_batch_size=16,
                eval_accumulation_steps=2,
                evaluation_strategy = "epoch",
                save_total_limit = 1,
                disable_tqdm = False,
                load_best_model_at_end=True,
                logging_steps=1,
                run_name = 'longformer-B'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=calc_classification_metrics,
        )

        print(f'Training model on fold {fold_count}')
        trainer.train()

        print(f'Validating model on fold {fold_count}')
        eval_data = trainer.evaluate()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=calc_classification_metrics,
        )

        print(f'Testing model on fold {fold_count}')
        test_data = trainer.evaluate()

        print('TEST RESULTS')
        print(test_data)

        evaluation[f'fold_{fold_count}'] = test_data

        remove(f'./cache/set_{set_num}')

        fold_count += 1

        torch.cuda.empty_cache()

    best_fold = find_best_fold(evaluation)
    print(f'Best fold is {best_fold}')
    to_remove = [fold for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'] if fold != best_fold]
    for i in to_remove:
        remove(f'./asap_aes_kfold/output/set_{set_num}_{percentage_of_training}_keyword/{i}')
    return evaluation

parser = argparse.ArgumentParser()

parser.add_argument('--set_num', type=int, default=3, help='set number')
parser.add_argument('--percentage_of_training', type=float, default=1, help='percentage of training data')
parser.add_argument('--label_list', default=[0, 1, 2, 3], nargs='+', type=int, help='label list')
parser.add_argument('--add_keywords', default=False)

args = parser.parse_args()

torch.cuda.empty_cache()

print(args.label_list)

if args.add_keywords:
    print('USING KEYWORDS')
    evaluation = train_model_with_keywords_kfold(args.set_num, args.percentage_of_training, freeze_bert=False, bert_base_path='', epochs=4, label_list=args.label_list)
    with open(f'./asap_aes_kfold/output/set{args.set_num}_{args.percentage_of_training}_keyword_eval.json', 'w') as f:
        json.dump(evaluation, f)
else:
    model, evaluation = train_model_kfold(args.set_num, args.percentage_of_training, args.label_list)
    with open(f'./asap_aes_kfold/output/set{args.set_num}_{args.percentage_of_training}_eval.json', 'w') as f:
        json.dump(evaluation, f)
