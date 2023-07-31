


temp = []
switch = False
with open("../dataset/textResults.txt", "r") as file1:
    for line in file1:
        if line.startswith("/home/umair/Desktop/factcheck/datasets/factbench/factbench"):
            switch = False
        if switch == True:
            continue
        if line.startswith("/home/umair/Desktop/factcheck/datasets/factbench/factbench"):
            if (temp.__contains__(line)):
                switch = True
                continue
            else:
                switch = False

        temp.append(line)

with open("../dataset/textResults2.txt", "w") as prediction_file:
    for line in temp:
        prediction_file.write(line)

print(len(temp))
print(temp)


#
#
#
# def load_data(data_dir, data_type):
#     try:
#         data = []
#         with open("%s%s.txt" % (data_dir, data_type), "r") as f:
#             for datapoint in f:
#                 datapoint = datapoint.split()
#                 if len(datapoint) == 4:
#                     s, p, o, label = datapoint
#                     if label == 'True':
#                         label = 1
#                     else:
#                         label = 0
#                     data.append((s, p, o, label))
#                 elif len(datapoint) == 3:
#                     s, p, label = datapoint
#                     assert label == 'True' or label == 'False'
#                     if label == 'True':
#                         label = 1
#                     else:
#                         label = 0
#                     data.append((s, p, 'DUMMY', label))
#                 else:
#                     raise ValueError
#     except FileNotFoundError as e:
#         print(e)
#         print('Add empty.')
#         data = []
#     return data
#
#
# def get_relations(data):
#     relations = sorted(list(set([d[1] for d in data])))
#     return relations
#
# def get_entities(data):
#     entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
#     return entities
#
#
#
# train_set = list(set(load_data(data_dir="dataset/", data_type="train")))
# test_set = list((load_data(data_dir="dataset/", data_type="test")))
#
# print(len(test_set))
#


