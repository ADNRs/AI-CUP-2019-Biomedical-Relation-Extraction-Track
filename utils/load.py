import pandas as pd

def load_dict(name):
    # load dictionary (one of the python data type) from the designated .txt file
    words = {}
    with open('./parameters/' + name + '.txt', 'r') as txt:
        for line in txt:
            key, val = line.split('|')
            if key.isnumeric():
                key = int(key)
            if val.split('\n')[0].isnumeric():
                val = int(val.split('\n')[0])
            else:
                val = val.split('\n')[0]
            words[key] = val
    return words

def load_data(name, pred = False, verbose = True):
    # load data from the name of the .tsv file
    if verbose:
        print('Loading data: ', end = '')
    name = './datasets/' + name
    data = pd.read_csv(name, sep = '\t')
    PMID = data['PMID'].to_numpy()
    sentence_id = data['Sentence_ID'].to_numpy()
    sentence = data['Sentence'].to_numpy()
    gene1 = data['Gene1|Gene1_ID'].to_numpy()
    gene2 = data['Gene2|Gene2_ID'].to_numpy()
    gene1_idx = data['Gene1_Index(start|end)'].to_numpy()
    gene2_idx = data['Gene2_Index(start|end)'].to_numpy()
    if verbose:
        print('Done')
    if pred:
        return PMID, sentence_id, sentence, gene1, gene2, gene1_idx, gene2_idx
    else:
        return PMID, sentence_id, sentence, gene1, gene2, gene1_idx, gene2_idx, data['RE_Type'].to_numpy()