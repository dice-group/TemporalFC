from rdflib import Graph, Namespace

# Define namespaces
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
ttv = Namespace("http://swc2017.aksw.org/")
# Load the Turtle data
# File paths
input_file = '/home/umair/Downloads/output-test_factbench.ttl'
output_file = '/home/umair/Downloads/output-test_factbench.nt'
# '<http://swc2017.aksw.org/task2/dataset/award_00000.ttl> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
# final_sentences = []
# with open(input_file, "r") as f:
#     for datapoint in f:
#         data = datapoint.split(" ")
#         sentence =
# Reading and processing the file in chunks of 5 lines
graph = Graph()
final_lines = ""
with open(input_file, 'r') as file:
    lines = file.readlines()
    n=0
    m=0
    for line in lines:
        line = line.replace(".ttl>","-"+str(m)+".ttl>")
        final_lines = final_lines+ line
        n=n+1
        if n%5==0:
            m =  m+1
    # for i in range(0, len(lines), chunk_size):
    #     chunk = lines[i:i + chunk_size]
    #     # process_lines(chunk)
    #     graph2 = Graph()
    #     graph2.parse(data=chunk, format='turtle')
    #     triples = graph2.triples((None, None, None))
    #     graph.add(triples)


graph.parse(data=final_lines, format='turtle')
# Load the Turtle data from file
# graph = Graph()
# with open(input_file, 'r') as f:
#     graph.parse(file=f, format='turtle')

subjects = set(graph.subjects(predicate=rdf.subject))
# Extract specific triples based on subject, predicate, and object
triples = graph.triples((None, None, rdf.Statement))
extracted_data = []
for subj, _, _ in triples:
    predicate = graph.value(subj, rdf.predicate)
    sub2 = graph.value(subj, rdf.subject)
    obj2 = graph.value(subj, rdf.object)
    ver = graph.value(subj, ttv.hasTruthValue)
    if predicate:
        extracted_data.append(f"<{sub2}> <{predicate}> <{obj2}> {ver} .\n")
    else:
        print("test issue here")

extracted_data = set(extracted_data)
# Write extracted triples in N-Triples format to a file
with open(output_file, 'w') as f:
    f.writelines(extracted_data)
