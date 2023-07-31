from urllib.request import urlopen
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, OWL
import os

def getSPO(g, countFb, cat):
    # s , p ,o = None
    for s, p, o in g.triples((None, URIRef('http://dbpedia.org/ontology/' + entityTopredicate(cat)), None)):
        # pprint.pprint(s + ' ' + p + ' ' + o)
        if (not (None, None, s) in g):
            print("error")
            print(g.triples(None, None, s))
            exit(1)
        for s1, p1, o1 in g.triples((None, None, s)):
            if str(s1).__contains__("freebase"):
                for s2, p2, o2 in g.triples((s1, OWL.sameAs, None)):
                    if str(o2).startswith("http://dbpedia.org/"):
                        s1 = o2
                        break
                    if str(o2).startswith("http://en.dbpedia.org/"):
                        s1 = o2
                        break
            if str(s1).__contains__("freebase"):
                countFb = countFb + 1
                print(s1)
                # s1 = "http://dbpedia.org/resource/" + sub.replace(" ", "_")
                return None
            s = s1
        for s1, p1, o1 in g.triples((None, None, o)):
            if str(o1).__contains__("freebase"):
                for s2, p2, o2 in g.triples((o1, OWL.sameAs, None)):
                    if str(o2).startswith("http://dbpedia.org/"):
                        o1 = o2
                        break
                    if str(o2).startswith("http://en.dbpedia.org/"):
                        o1 = o2
                        break
            if str(o1).__contains__("freebase"):
                countFb = countFb + 1
                print(o1)
                return None
                # o1 = "http://dbpedia.org/resource/" + obj.replace(" ", "_")
            o = o1
    return s, p, o
# class step1:
def entityTopredicate(predicate):
    p = predicate
    if p == "birth":
        p = "birthPlace"
    if p == "death":
        p = "deathPlace"
    if p == "foundationPlace":
        p = "foundationPlace"
    if p == "starring":
        p = "starring"
    if p == "award":
        p = "award"
    if p == "subsidiary":
        p = "subsidiary"
    if p == "publicationDate":
        p = "author"
    if p == "spouse":
        p = "spouse"
    if p == "leader":
        p = "office"
    if p == "nbateam":
        p = "team"
    return p

countFb =0
fact_bench_path = "/home/umair/Desktop/factcheck/datasets/factbench-2/"
g = Graph()
datasets= ['train/', 'test/']
dataset_label = ['correct/', 'wrong/']
# true_cat = ['award/', 'birth/', 'death/', 'foundationPlace/','leader/','publicationDate/', 'spouse/', 'starring/','subsidiary/']
# false_cat = ['domain/','domainrange/','mix/','property/','random/','range/']
for data in datasets:
    for lbl in dataset_label:
        if lbl.__eq__('correct/'):
            for cat in os.listdir(fact_bench_path + data + lbl):
                for entry in os.listdir(fact_bench_path + data + lbl + cat):
                    g.parse(fact_bench_path + data + lbl + cat +'/'+ entry)
                    print(fact_bench_path + data + lbl + cat +'/'+ entry)
                    try:
                        s,p,o =  getSPO(g,countFb,cat)
                        print(s + '-' + p + '-'+ o )
                    except:
                        print("s")
        else:
            for cat in os.listdir(fact_bench_path + data + lbl):
                for cat1 in os.listdir(fact_bench_path + data + lbl + cat):
                    for entry in os.listdir(fact_bench_path + data + lbl + cat+'/'+ cat1 ):
                        if not os.path.isdir(fact_bench_path + data + lbl + cat +'/'+ cat1 +'/'+ entry):
                            g.parse(fact_bench_path + data + lbl + cat +'/'+ cat1 +'/'+ entry)
                            print(fact_bench_path + data + lbl + cat +'/'+ cat1 +'/'+ entry)
                            try:
                                s, p, o = getSPO(g,countFb,cat1)
                                print(s + ' -' + p + ' -' + o)
                            except:
                                print("s")

                            # import pprint
                            # for stmt in g:
                            #     pprint.pprint(stmt)
                        else:
                            for entry2 in os.listdir(fact_bench_path + data + lbl + cat +'/'+ cat1+'/'+ entry):
                                g.parse(fact_bench_path + data + lbl + cat +'/'+ cat1 +'/'+ entry + '/'+entry2)
                                print(fact_bench_path + data + lbl + cat +'/'+ cat1 +'/'+ entry + '/' +entry2)
                                try:
                                    s, p, o = getSPO(g,countFb,cat1)
                                    print(s + ' -' + p + ' -' + o)
                                except:
                                    print("s")


import pprint
for stmt in g:
    pprint.pprint(stmt)

exit(1)
sparql = SPARQLWrapper("http://sparql.cs.upb.de:8891/sparql")
text = ""
query = """PREFIX rdfs: http://www.w3.org/2000/01/rdf-schema#
    PREFIX wd: http://www.wikidata.org/entity/
    SELECT ?item
    WHERE {
    wd:"""+1+""" rdfs:label ?item .
    FILTER (langMatches( lang(?item), "EN" ) )
    }
    LIMIT 1 """
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
text = results['results']['bindings'][0]['item']['value']
