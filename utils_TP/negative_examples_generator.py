import numpy as np

class NegativeExamplesGenerator:
    def __init__(self, n_predicates, preicates):
        self.n_predicates = n_predicates
        self.predicates = preicates

    def get_keys_from_value(self, dictionary, target_value):
        keys = [key for key, value in dictionary.items() if value == target_value]
        return int(keys[0])
    def generate_negative_triples(self, quadruples):
        new_quadruples = []
        i = 0
        for quad in quadruples:
            tt = (self.get_keys_from_value(self.predicates,quad[1])) + self.n_predicates // 2
            tt = tt % 7
            new_quadruples.append([quad[2],self.predicates.get(tt),quad[0],'False'])
        # Make a copy of the original examples
        # copy = np.copy(quadruples)
        #
        # # Swap head and tail entities
        # tmp = np.copy(copy[:, 0])
        # copy[:, 0] = copy[:, 2]
        # copy[:, 2] = tmp
        #
        # # Modify relation
        # copy[:, 1] += self.n_predicates // 2

        return quadruples+new_quadruples

def save_relations_as_dict(quadruples):
    distinct_relations = list(set(quad[1] for quad in quadruples))
    relations_dict = {i: rel for i, rel in enumerate(distinct_relations)}
    return relations_dict
def read_quadruples_from_file(file_path):
    quadruples = []
    with open(file_path, 'r') as file:
        for line in file:
            quadruple = line.strip().split('\t')  # Assuming tab-separated values
            quadruples.append(quadruple)
    return quadruples
def write_quadruples_to_file(quadruples, file_path):
    with open(file_path, 'w') as file:
        for quad in quadruples:
            file.write('\t'.join(quad) + '\n')

data_type = 'test'
file_path = '/home/umair/Documents/pythonProjects/TemporalFC/data_TP/dbpedia124k/'+data_type+'/'+data_type+'_with_time_final'  # Replace with the actual path to your file

# Read quadruples from the file
quadruples = read_quadruples_from_file(file_path)

distinct_relations = save_relations_as_dict(quadruples)

# Example usage
n_predicates = len(distinct_relations)  # Replace with the actual number of predicates
data_processor = NegativeExamplesGenerator(n_predicates,preicates=distinct_relations)
# Assuming 'examples' is your original array of quadruple
# Generate negative triples
negative_triples = data_processor.generate_negative_triples(quadruples)

write_quadruples_to_file(negative_triples,'/home/umair/Documents/pythonProjects/TemporalFC/data_TP/dbpedia124k/'+data_type+'/'+data_type+'_final.txt')
