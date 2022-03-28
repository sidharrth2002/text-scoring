import nlpaug.augmenter.word as naw
import nlpaug.flow.sequential as naf
import pandas as pd

# link file below
data = pd.read_csv('')
scores_to_augment_and_ratio = {
    # total 400 extra
    2: {
        "amount": 200,
        "multiplier": 2
    },
    # total 900 extra
    3 : {
        "amount": 300,
        "multiplier": 3
    }

}

ACT = 'substitute'

aug_sy = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', lang='eng', 
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False, 
                     verbose=0)

aug_bert = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', 
    action=ACT)

# BERT seq-to-seq followed by synonym replacement
aug = naf.Sequential([
    aug_bert,aug_sy
])

for i in scores_to_augment_and_ratio.keys():
    augmented = []
    for i in data[data['user_score'] == i]['text'].sample(scores_to_augment_and_ratio[i]['amount']):
        augmented_text = aug.augment(i, n=scores_to_augment_and_ratio[i]['multiplier'])
        print(augmented_text)
        augmented += augmented_text

    augmented_df = pd.DataFrame({'text': augmented, 'user_score': i, 'applied': 1, 'length': 0})

    data = pd.concat([data, augmented_df])