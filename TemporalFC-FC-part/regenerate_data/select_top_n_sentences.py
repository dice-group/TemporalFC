import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def add_edges(G,p):
    for i in G.nodes():
        for j in G.nodes():
            if i!=j:
                r = random.random()
                if r<p:
                    G.add_edge(i,j)
                else:
                    continue
    return G

def distribute(G, prev_points):
    new_points = [0 for i in range(G.number_of_nodes())]
    for i in G.nodes():
        out = G.out_edges(i)
        if len(out)==0:
            new_points[i] += prev_points[i]
        else:
            share = prev_points[i] / len(out)
            for each in out:
                new_points[each[1]] += share
    return G, new_points



def initialize(G):
    points = [100 for i in range(G.number_of_nodes())]
    return points

def make_converge(G,points):
    prev_points = points
    print('type # to stop the code')
    while(1):
        G, new_points = distribute(G,prev_points)
        print(new_points)
        char = input()
        if char == '#':
            break
        prev_points = new_points
    return G, new_points



def get_nodes_rank(points):
    np_array = np.array(points)
    rank_index = np.argsort(-np_array)
    return rank_index




def main():
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(10)])
    G = add_edges(G,0.3)
    points = initialize(G)
    # distribution = distribute(G,points)
    G, points = make_converge(G,points)
    # return converge
    node_rank = get_nodes_rank(points)
    print(node_rank)

    page_ranke = nx.pagerank(G)
    rank_of_nodes = sorted(page_ranke.items(), key = lambda x:x[1], reverse=True)
    for i in rank_of_nodes:
        print(i[0],)

    print(rank_of_nodes)
    return node_rank

    # return nx.draw(G), points
    # return distribution
from io import BytesIO, StringIO
# import os
# from io import BytesIO
# import matplotlib.pyplot as plt
# # import rawpy
# import exifread
# import numpy as np
# from PIL import Image
import codecs, csv
import operator
def select_top_n_sentences(sentences, n, stncs,type, propr=None):
    print("selecting top n sentences")
    sens_web = dict()
    i = 0
    for st in zip(sentences,stncs):
        if st[0] in sens_web.keys():
            sens_web[st[0]+''+str(i)] = st[1]
            i += 1
        else:
            sens_web[st[0]] = st[1]
    # sens_web = zip(sentences,stncs)
    sens = {}
    flag = False
    i = 0
    pth = ''
    if propr == None:
        pth = '../dataset/pg_ranks/all_websites_ids_'+type+'_pagerank.txt'
    else: #ids_author_all_entities all_websites_ids_bpdp_complete_pagerank
        pth = '../dataset/pg_ranks/all_websites_ids_pagerank.txt'
    for s in sentences:
        s = s.replace(" ","_")
        with open(pth, 'r') as f:
            for xx in f:
                if flag:
                    if s in sens.keys():
                        sens[s+''+str(i)] = xx[:-1]
                        i +=1
                    else:
                        sens[s] = xx[:-1]

                    break
                if xx[:-1] == s:
                    print('found->',xx)
                    flag = True
        if not flag:
            print('Not found->', s)
            # exit(1)
        else:
            flag = False


    sorted_d = dict(sorted(sens.items(), key=operator.itemgetter(1), reverse=True))
    print('Dictionary in descending order by value : ', sorted_d)
    final_sentences = []
    for sk in sorted_d:
        final_sentences.append(sens_web[sk])

    return list(sorted_d.items())[:n], final_sentences[:n]





if __name__ == '__main__':
    with codecs.open('/home/umair/Downloads/pg_rank/wikipedia-pagerank-page-id-title.raw', 'rb', 'utf-8') as f:
        # xx= f.readline()
        for xx in f:
            print(xx)
        # xx = BytesIO(xx)

        # try:
        #     text_stream = StringIO(xx.getvalue())
        #     print(text_stream)
        # except TypeError:
        #     print('Sorry, text stream cannot store bytes')
        # # print(StringIO(xx.getvalue()))
# read from f

    # print(main())
    # with rawpy.imread('/home/umair/Downloads/pg_rank/wikipedia-pageranks.raw','rb', buffering=0) as reader:
    #     print(reader.read())
    # A = np.fromfile('/home/umair/Downloads/pg_rank/wikipedia-pageranks.raw', dtype='int16', sep="")
    # A = A.reshape([1024, 1024])
    # plt.imshow(A)
    # with open('/home/umair/Downloads/pg_rank/wikipedia-pageranks.raw', 'rb') as f:
    #     binary_data = f.readline()
    #     text = binary_data.decode('latin-1')
    #     print(text)
        # tags = exifread.process_file(f)
        # for key, value in tags.items():
        #     if key != 'JPEGThumbnail':  # do not print (uninteresting) binary thumbnail data
        #         print(f'{key}: {value}')

