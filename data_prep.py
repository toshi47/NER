import pandas as pd
from nerus import load_nerus

sent_id_list = []
word_list = []
word_id_in_sent_list = []
part_of_speech_list = []
feats_list = []
head_id_list = []
rel_list = []
tag_list = []

def data_init():
    stop_iteration_id = 739345
    docs = load_nerus("nerus_lenta.conllu.gz")
    #print(type(docs))
    doc = next(docs)
    while (doc.id != '1000'):   #next(docs) != StopIteration
        print("doc id - ", doc.id)
        sent = doc.sents
        for i in range(len(sent) - 1):
            # print(sent[i])
            pos=int(sent[i].id.find('_'))
            #print('sent id - ', pos)
            #print('full - ',sent[i].id)
            sent_id = int(doc.id) + int(sent[i].id[pos+1:])
            #print('int - ',sent_id)
            tokens = sent[i].tokens
            for j in range(len(tokens) - 1):
                nerus_token = tokens[j]
                word_id_in_sent = nerus_token.id
                word = nerus_token.text
                part_of_speech = nerus_token.pos
                feats = nerus_token.feats
                head_id = nerus_token.head_id
                rel = nerus_token.rel
                tag = nerus_token.tag
                word_list.append(word)
                word_id_in_sent_list.append(word_id_in_sent)
                part_of_speech_list.append(part_of_speech)
                feats_list.append(feats)
                head_id_list.append(head_id)
                rel_list.append(rel)
                tag_list.append(tag)
                sent_id_list.append(sent_id)
        doc = next(docs)


data_init()
d = {'word': word_list, 'sent_id': sent_id_list, 'word_id_in_sent': word_id_in_sent_list,
     'part_of_speech': part_of_speech_list, 'head_id': head_id_list, 'rel': rel_list}
df = pd.DataFrame(data=d)
df = pd.get_dummies(df, columns=['part_of_speech', 'rel'], drop_first=True)

feats_df = pd.DataFrame(feats_list)

feats_df = feats_df.fillna(0)

for column in feats_df.columns:
    dummies = pd.get_dummies(feats_df[column], prefix=column)
    feats_df = pd.concat([feats_df, dummies], axis=1)
    feats_df = feats_df.drop(columns=column)

joined_df_merge = df.merge(feats_df, how='left',
                           left_index=True,
                           right_index=True)

joined_df_merge.insert(len(joined_df_merge.columns), 'tag', tag_list)

print(joined_df_merge.columns)
num_of_classes = len(joined_df_merge['tag'].unique())
print(num_of_classes)

joined_df_merge.to_csv("my_small.csv")


# print(type(doc))
