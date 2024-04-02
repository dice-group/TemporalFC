
entities = set()
with open("/data_TP/factbench/all_entities.txt", "r") as file:
    for line in file:
        entities.add(line[:-1])

print(len(entities))
embeddings = []


with open("/home/umair/Desktop/NEBULA/TransE_entity_embeddings.csv", "r") as file:
    for line in file:
        # print(line)
        emb = line.split(">")[0]+">"
        if emb in entities:
            embeddings.append(emb)



print(len(embeddings))


