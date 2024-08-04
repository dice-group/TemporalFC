typ = 'train'
path = '/local/upb/users/u/uqudus/profiles/unix/cs/TemporalFC/TemporalFC/data_TP/dbpedia124k/'
data_dir = path+typ+'/'
with open(data_dir+typ+'2.txt', 'r') as input_file, open(data_dir+typ+'3', 'w') as output_file:
    for line in input_file:
        parts = line.strip().split('\t')
        # Extract the last part after '/' and '>' from each field
        subject = parts[0].split('/')[-1].rstrip('>')
        predicate = parts[1].split('/')[-1].rstrip('>')
        obj = parts[2].split('/')[-1].rstrip('>')
        year = parts[3]
        boolean_value = parts[4]

        # Write the modified line to the output file
        output_line = f"{subject}\t{predicate}\t{obj}\t{year}\t{boolean_value}\n"
        output_file.write(output_line)
