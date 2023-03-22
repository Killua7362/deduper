import recordlinkage
from nltk import word_tokenize
from nltk.tokenize import TreebankWordDetokenizer
import pandas as pd
import numpy as np
from recordlinkage.datasets import load_febrl2
from recordlinkage.index import Block
from recordlinkage.index import SortedNeighbourhood
from sklearn import model_selection
from xgboost import XGBClassifier
import xgboost as xgb
#Load Freely Extensible Biomedical Record Linkage data
df = load_febrl2()
#converting all string to uppercase
df = df.astype(str).apply(lambda x:x.str.upper())
#postcode cleanup
df['postcode'] = df['postcode'].str.strip()
df['postcode'] = df['postcode'].str.findall('[0-9]+')
df['postcode'] = df['postcode'].str.join("")
df['postcode'] = df['postcode'].fillna("")

named_stopwords = ['STREET','ST','PLACE','RD','WORD']
df['address_1_token'] = df['address_1'].apply(word_tokenize)
df['address_1'] = df['address_1_token'].apply(lambda x:[word for word in x if word not in named_stopwords])
df['address_1'] = df['address_1'].apply(TreebankWordDetokenizer().detokenize)
df['address_1'] = df['address_1'].str.replace("[\'\".,()*+&\/\-\\\+\!\%:;?]"," ")
df['address_2'] = df['address_2'].str.replace("[\'\".,()*+&\/\-\\\+\!\%:;?]"," ")

Block_Index_by_State = Block(on="state")
Block_Index_by_State_Pairs = Block_Index_by_State.index(df)
Neighbour_index_by_name = SortedNeighbourhood(on='surname',window=5)
Neighbour_index_by_name_pairs = Neighbour_index_by_name.index(df)
all_index_pairs = Block_Index_by_State_Pairs.append(Neighbour_index_by_name_pairs)
all_index_pairs = all_index_pairs.drop_duplicates(keep='first')

compare = recordlinkage.Compare()
compare.exact('given_name', 'given_name', label='given_name')
compare.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
#compare.date('date_of_birth', 'date_of_birth', label='date_of_birth')
compare.exact('suburb', 'suburb', label='suburb')
compare.exact('state', 'state', label='state')
compare.string('address_1', 'address_1', threshold=0.85, label='address_1')


comparison_vectors = compare.compute(all_index_pairs, df)
#eatures[features.sum(axis=1) > 3]
df, links = load_febrl2(return_links=True)
duplicate_pairs_vectors = compare.compute(links,df)

duplicate_pairs = duplicate_pairs_vectors.reset_index()
duplicate_pairs_1 = duplicate_pairs["level_0"]+','+duplicate_pairs["level_1"]
duplicate_pairs_2 = duplicate_pairs["level_1"]+','+duplicate_pairs["level_0"]
final_duplicate_pairs = pd.DataFrame(duplicate_pairs_1.append(duplicate_pairs_2))
comparison_pairs = comparison_vectors.reset_index()
comparison_pairs['join_keys'] = comparison_pairs["rec_id_1"]+','+comparison_pairs["rec_id_2"]
comparison_pairs['Label'] = np.where(comparison_pairs["join_keys"].isin(final_duplicate_pairs[0]),"1","0")

Model_data_set = comparison_pairs.set_index(['join_keys','rec_id_1','rec_id_2'])
Model_data_set['Label'] = Model_data_set['Label'].astype(int)
y= Model_data_set.Label
x= Model_data_set.drop(['Label'],axis=1)

test_size = 0.4
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=test_size, random_state=42, stratify=y)

model= xgb.XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model.fit(x_train, y_train)

y_pred = pd.DataFrame(model.predict(x_test))
predictions = y_pred
predictions['predict'] = y_pred
dfcombine = pd.merge(x_test.reset_index(),predictions[['predict']],how='left',left_index=True,right_index=True)

