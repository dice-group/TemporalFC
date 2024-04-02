import pandas as pd
# If we have entities missing in the embeddings, we need to remove them from the train and test triples because we can
# procceed with incomplete embeddings set
# Step 1: Read embeddings.csv and store entities in a set
type = 'train'
data_set = 'factbench'
embeddings_file_path = '../data_TP/factbench/embeddings/TransE/all_entities_embeddings_final.csv'
embeddings_df = pd.read_csv(embeddings_file_path, header=None)
entities_set = set(embeddings_df[0])

# Step 2: Read train.txt and get all subject and object entities
train_file_path = '../data_TP/'+data_set+'/'+type+'/'+type
train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['subject', 'predicate', 'object', 'v','dot'])
all_entities = set(train_df['subject']).union(set(train_df['object']))

# Step 3: Remove triples with missing entities in embeddings set
filtered_train_df = train_df[(train_df['subject'].isin(entities_set)) & (train_df['object'].isin(entities_set))]

# Print the resulting DataFrame or save it to a new file if needed
print(filtered_train_df)
# Store the filtered_train_df in another text file separated by a specific delimiter
output_file_path = train_file_path+'1'
filtered_train_df.to_csv(output_file_path, sep='\t', index=False, header=None)

print(f"Filtered DataFrame saved to {output_file_path}")
