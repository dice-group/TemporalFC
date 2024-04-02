# Open the file
import json
import re



def add_backslash_to_quotes(input_string):
    input_string = input_string.replace("\xad","")
    input_string = input_string.replace("\xa0","")
    input_string = repr(input_string)
    if input_string[0] != "\"" and input_string[-1] != "\"":
        input_string = input_string.replace('"', '\\"')
        input_string = "\"" + input_string + "\""
    # input_string = input_string.replace("\\\\x","\\x")
    input_string = input_string.replace("\\\'", "\'")
    input_string = input_string.replace("\\u200e", "\u200e")
    return input_string

dataset_path = '/home/umair/Documents/pythonProjects/testt/dbpedia124k/dbpedia124k/'
typ = 'test'

with open(dataset_path+typ+'/'+typ+'.txt', 'r') as file:
    lines = file.readlines()

with open(dataset_path+typ+'/result_FactCheck_'+typ+'_FactCheck.txt', 'r') as file:
    sent_lines = file.readlines()

final_sent_lines = []
triples_with_sentences = {}
final_triples = []
for line in sent_lines:
    if line.startswith('{'):
        final_sent_lines.append(line)
        json_obj = json.loads(line)
        final_triples.append('<' + json_obj['subject'] + '>' + '\t' + '<' + json_obj['predicate'] + '>' + '\t' + '<' + json_obj[
                'object'] + '>')
        triples_with_sentences['<' + json_obj['subject'] + '>' + '\t' + '<' + json_obj['predicate'] + '>' + '\t' + '<' + json_obj[
                'object'] + '>'] = json_obj['complexProofs']

# for line in final_sent_lines:

# Create dictionaries to store True and False lines
true_lines = {}
false_lines = {}
trues_lines = []
# Parse the lines and store True and False lines separately
for line in lines:
    data = line.strip().split('\t')
    if len(data)==5:
        if data[4] == 'True':
            true_lines[(data[0], data[2])] = data[3]
            if ('\t'.join(data[:3]) in final_triples):
                trues_lines.append('\t'.join(data))
        else:
            false_lines[(data[0], data[2])] = data
    elif len(data)==4:
        if data[3] == 'False':
            false_lines[(data[0], data[2])] = data
        else:
            print("check here:"+ data)
            exit(1)
files_lines = []
# Copy year from True lines to corresponding False lines
for key, value in false_lines.items():
    reverse_key = (key[1], key[0])  # Swap <s> and <o>
    if reverse_key in true_lines:
        value[3] = true_lines[reverse_key]
        if ('\t'.join(value[:3]) in final_triples):
            files_lines.append('\t'.join(value)+'\tFalse')

    # Write the updated data back to the file
sent_final = []
with open(dataset_path+typ+'/'+typ+'2.txt', 'w') as output_file:
    for line in set(trues_lines):
        output_file.write(line + '\n')
        sent_final.append(triples_with_sentences['\t'.join(line.split('\t')[:3])])
    for line in set(files_lines):
        output_file.write(line+ '\n')
        sent_final.append(triples_with_sentences['\t'.join(line.split('\t')[:3])])
with open(dataset_path + typ + '/' + typ + '_sentences.txt','w') as output_file:
    for line in sent_final:
        if len(line)==0:
            output_file.write('{}\n')
        else:
            for ll in line:
                # json_obj = json.loads(ll['proofPhrase'])
                output_file.write('{\"proofPhrase\":'+ add_backslash_to_quotes(ll['proofPhrase']) + ', \"pagerank\":\"'+str(ll['pagerank']) + '\"}\t')

            output_file.write('\n')
print("Year data updated for False lines where corresponding True lines were found. also saved as: " + dataset_path + typ + ' sentences.txt')
