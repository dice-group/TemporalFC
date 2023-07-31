from multiprocessing import Process
import sys
import re
# first step is to find the unique triples by applying following command
# sort filtered_dbpedia_03_2021.nt | uniq -u > distinct_filtered_dbpedia_03_2021.nt
def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)

    path = sys.argv[1:][0]
    file_main = sys.argv[1:][1]
    # /home/umair/Documents/filteringKG/
    i=0
    lines = []
    infered_lines = []
    symmetric_count = 0
    inverse_count = 0
    print("start of first phase: symmetric and inverse property")
    # clearing all files
    with open(path + "inferred_dbpedia_03_2021.nt", 'w') as writer:
        print ("")
#    with open(path + "subClassPredicates.nt", 'w') as writer:
#        print("")
#    with open(path + "subPropertyPredicates.nt", 'w') as writer:
#        print("")
#    with open(path + "subRegionPredicates.nt", 'w') as writer:
#        print("")
#    with open(path+file_main,'r') as writer:
#        for line in writer:
#            s1= line.split(" ")[0]
#            s2= line.split(" ")[1]
#            s3= line.split(" ")[2]
#            if ("/ontology/parent>" or "/ontology/child>" or "/ontology/mother>" or "/ontology/father>") in s2:
#                print(s2)
#                print(line)
#                if "/ontology/parent" in s2:
#                    s2 = s2.replace("/ontology/parent","/ontology/child")
#                elif "/ontology/child" in s2:
#                    s2 = s2.replace("/ontology/child","/ontology/parent")

#                if "/ontology/mother" in s2:
#                    s2 = s2.replace("/ontology/mother","/ontology/child")

#                if "/ontology/father" in s2:
#                    s2 = s2.replace("/ontology/father","/ontology/child")

#                infered_lines.append(s3 + " " + s2 + " " + s1 + " .\n")
#                inverse_count = inverse_count+1
#                if len(infered_lines) >= 100000:
#                    with open(path + "inferred_dbpedia_03_2021.nt", 'a') as writer:
#                        for l2 in infered_lines:
#                    infered_lines.clear()


#            if ("/ontology/spouse>" or "/ontology/friend>" or "/ontology/sibling>") in s2:
#                print(s2)
#                print(line)

#                infered_lines.append(s3 + " " +s2 +" " +s1 + " .\n")
#                symmetric_count = symmetric_count+1
#                if len(infered_lines)>=100000:
#                    with open(path + "inferred_dbpedia_03_2021.nt", 'a') as writer:
#                        for l2 in infered_lines:
#                            writer.write(l2)
#                    infered_lines.clear()
#        if len(infered_lines)!=0:
#            with open(path + "inferred_dbpedia_03_2021.nt", 'a') as writer:
#                for l2 in infered_lines:
#                    writer.write(l2)
#            infered_lines.clear()



#        print ("test")

    # apply following commands first
    # grep - i " <http://www.w3.org/2000/01/rdf-schema#subPropertyOf> "  distinct_filtered_dbpedia_03_2021.nt > subPropertyPredicates.nt
    # grep - i " <http://www.w3.org/2000/01/rdf-schema#subClassOf> " distinct_filtered_dbpedia_03_2021.nt > subClassPredicates.nt
    # grep - i " <http://dbpedia.org/ontology/subregion> " distinct_filtered_dbpedia_03_2021.nt > subRegionPredicates.nt
    print("start of second phase: transitive property")

    # collecting all transitive properties -> subclass, subproperty and subregion
    transitive_count = 0
    sub_class_prop = []
    sub_property_prop = []
    sub_region_prop = []
    inferred_lines_subclass = []
    inferred_lines_subProperty = []
    inferred_lines_subRegion = []
    with open(path+file_main, 'r') as f:
        for line in f:
            if "> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <" in line:
                print (line)
#                sub_class_prop.append(line)
                inferred_lines_subclass.append(line)
            if "> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf> <" in line:
#                print (line)
#                sub_property_prop.append(line)
                inferred_lines_subProperty.append(line)
#            if "> <http://dbpedia.org/ontology/subregion> <" in line:
#                print (line)
#                sub_region_prop.append(line)
#                inferred_lines_subRegion.append(line)

            if len(sub_class_prop)>=100000:
                with open(path+"subClassPredicates.nt",'a') as writer:
                    for l2 in sub_class_prop:
                        writer.write(l2)
                sub_class_prop.clear()
            if len(sub_property_prop)>=100000:
                with open(path+"subPropertyPredicates.nt",'a') as writer:
                    for l2 in sub_property_prop:
                        writer.write(l2)
                sub_property_prop.clear()
            if len(sub_region_prop)>=100000:
                with open(path+"subRegionPredicates.nt",'a') as writer:
                    for l2 in sub_region_prop:
                        writer.write(l2)
                sub_region_prop.clear()

    if len(sub_class_prop) > 0:
        with open(path + "subClassPredicates.nt", 'a') as writer:
            for l2 in sub_class_prop:
                writer.write(l2)
        sub_class_prop.clear()
    if len(sub_property_prop) > 0:
        with open(path + "subPropertyPredicates.nt", 'a') as writer:
            for l2 in sub_property_prop:
                writer.write(l2)
        sub_property_prop.clear()

    if len(sub_region_prop) > 0:
        with open(path + "subRegionPredicates.nt", 'a') as writer:
            for l2 in sub_region_prop:
                writer.write(l2)
        sub_region_prop.clear()

    # with open(path + file_main, 'r') as f:
    for line in inferred_lines_subclass:
        print(line)
        #
            # subclass inference
        if "> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <" in line:
            print(line)

                # s1 = line.split(" ")[0]
                # s2 = line.split(" ")[1]
                # s3 = line.split(" ")[2]
            if "> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <" in line:
                    #    print (line)

                s1 = line.split(" ")[0]
                s2 = line.split(" ")[1]
                s3 = line.split(" ")[2]
                    #
                    # file = open(path+"subClassPredicates.nt", "r")
                for line2 in inferred_lines_subclass:
                    if line2 != line:
                        s11 = line2.split(" ")[0]
                        s22 = line2.split(" ")[1]
                        s33 = line2.split(" ")[2]
                        if (s1 == s33):
                            print(line2)
                            transitive_count = transitive_count + 1
                            infered_lines.append(s11 + " " + s22 + " " + s3 + " .\n")
                        elif (s3 == s11):
                            print(line2)
                            transitive_count = transitive_count + 1
                            infered_lines.append(s1 + " " + s22 + " " + s33 + " .\n")
            # subproperty inference
            #            if "> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf> <" in line:
            #                print(line)

            #                s1 = line.split(" ")[0]
            #                s2 = line.split(" ")[1]
            #                s3 = line.split(" ")[2]

            # file = open(path + "subPropertyPredicates.nt", "r")
            #                for line2 in inferred_lines_subProperty:
            #                    if line2 != line:
            #                        s11 = line2.split(" ")[0]
            #                        s22 = line2.split(" ")[1]
            #                        s33 = line2.split(" ")[2]
            #                        if (s1 == s33):
            #                            print(line2)
            #                            transitive_count = transitive_count + 1
            #                            infered_lines.append(s11 + " " + s22 + " " + s3 + " .\n")
            #                        elif (s3 == s11):
            #                            print(line2)
            #                            transitive_count = transitive_count + 1
            #                            infered_lines.append(s1 + " " + s22 + " " + s33 + " .\n")

            # subregion inference
            #            if "> <http://dbpedia.org/ontology/subregion> <" in line:
            #                print(line)
            #                s1 = line.split(" ")[0]
            #                s2 = line.split(" ")[1]
            #                s3 = line.split(" ")[2]

            # file = open(path + "subRegionPredicates.nt", "r")
            #                for line2 in inferred_lines_subRegion:
            #                    if line2 != line:
            #                        s11 = line2.split(" ")[0]
            #                        s22 = line2.split(" ")[1]
            #                        s33 = line2.split(" ")[2]
            #                        if (s1 == s33):
            #                            # print(line2)
            #                            transitive_count = transitive_count + 1
            #                            infered_lines.append(s11 + " " + s22 + " " + s3 + " .\n")
            #                        elif (s3 == s11):
            #                            # print(line2)
            #                            transitive_count = transitive_count+1
            #                            infered_lines.append(s1 + " " + s22 + " " + s33 + " .\n")
            #                # lines.append(line)

            if len(infered_lines) >= 100000:
                with open(path + "inferred_dbpedia_03_2021.nt", 'a') as writer:
                    for l2 in infered_lines:
                        writer.write(l2)
                infered_lines.clear()

    if len(infered_lines)>0:
        with open(path + "inferred_dbpedia_03_2021.nt", 'a') as writer:
            for l2 in infered_lines:
                writer.write(l2)
        infered_lines.clear()

    exit(1)
    new_lines = []
    total_count = 0
    with open(path + "inferred_dbpedia_03_2021.nt", 'r') as f:
        for line in f:
            total_count = total_count +1
            print (line)
            new_lines.append(line)
            if len(new_lines) > 100000:
                with open(path+file_main,'a') as writer:
                    for l2 in new_lines:
                        writer.write(l2)
                new_lines.clear()
    if len(new_lines) > 0:
        with open(path + file_main, 'a') as writer:
            for l2 in new_lines:
                writer.write(l2)
        new_lines.clear()

    print ("total newly inferred statements added: "+ str(total_count))
    print ("total newly inferred transitive statements added: "+ str(transitive_count))
    print ("total newly inferred inverse statements added: "+str(inverse_count))

def par_process(line, inferred_lines_subclass, infered_lines, transitive_count):
    if "> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <" in line:
         #    print (line)

        s1 = line.split(" ")[0]
        s2 = line.split(" ")[1]
        s3 = line.split(" ")[2]
            #
        # file = open(path+"subClassPredicates.nt", "r")
        for line2 in inferred_lines_subclass:
             if line2 != line:
                 s11 = line2.split(" ")[0]
                 s22 = line2.split(" ")[1]
                 s33 = line2.split(" ")[2]
                 if (s1 == s33):
                     print(line2)
                     transitive_count = transitive_count + 1
                     infered_lines.append(s11 + " " + s22 + " " + s3 + " .\n")
                 elif (s3 == s11):
                     print(line2)
                     transitive_count = transitive_count + 1
                     infered_lines.append(s1 + " " + s22 + " " + s33 + " .\n")

if __name__ == "__main__":
    main()

    # print ("total newly inferred symmetric statements added: "+str(transitive_count))
