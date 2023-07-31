
import sys


def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)
    # /home/umair/Documents/filteringKG/
    path = sys.argv[1:][0]
    # "dbpedia_03_2021.nt"
    file_main1 = sys.argv[1:][1]
    file_main2 = sys.argv[1:][2]
    file_main3 = sys.argv[1:][3]

    i=0
    with open(path+"filtered_dbpedia_03_2021.nt",'w') as writer:
        print ("test")
    subj = []
    lines = []
    with open(path + file_main1, 'r') as f:
        triple = ""
        for line in f:
            print(line)
            s1 = line.split(" ")[0]
            s2 = line.split(" ")[1]
            s3 = line.split(" ")[2]
            if (s2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>"):
                if (s1 not in subj):
                    subj.append(s1)
                    triple = s3

            if (s2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>"):
                if (s1 in subj):
                    triple = triple+" "+s3

            if (s2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>"):
                if (s1 in subj):
                    triple = triple+" "+s3 +' .\n'
                    if (triple != ""):
                        lines.append(triple)
                        triple = ""

    subj = []
    lines2 = []
    with open(path + file_main2, 'r') as f:
        triple = ""
        for line in f:
            print(line)
            s1 = line.split(" ")[0]
            s2 = line.split(" ")[1]
            s3 = line.split(" ")[2]
            if (s2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>"):
                if (s1 not in subj):
                    subj.append(s1)
                    triple = s3

            if (s2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>"):
                if (s1 in subj):
                    triple = triple + " " + s3

            if (s2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>"):
                if (s1 in subj):
                    triple = triple + " " + s3 + ' .\n'
                    if (triple != ""):
                        lines2.append(triple)
                        triple = ""
    # with open(path+"test2.nt",'w') as writer:
    #     writer.writelines(lines)
    # with open(path+"train2.nt",'w') as writer:
    #     writer.writelines(lines2)
    for l in lines:
        print(l)
    final_lines = []
    with open(path + file_main3) as f:
        for line in f:
            if (line not in lines) and (line not in lines2):
                print(line)
                final_lines.append(line)

            if len(final_lines) >= 100000:
                with open(path + "filtered_dbpedia_03_2021.nt", 'a') as writer:
                    for l2 in final_lines:
                        writer.write(l2)
                final_lines.clear()
    if len(final_lines) >= 0:
        with open(path + "filtered_dbpedia_03_2021.nt", 'a') as writer:
            for l2 in final_lines:
                writer.write(l2)
        final_lines.clear()


    # with open(path+file_main) as f:
    #     for line in f:
    #         # print (line)
    #         i =  i +1
    #         if "\"" not in line and "dbpedia.org" in line:
    #             print (line)
    #             lines.append(line)
    #         if i%100000==0:
    #             with open(path+"filtered_dbpedia_03_2021.nt",'a') as writer:
    #                 for l2 in lines:
    #                     writer.write(l2)
    #             lines.clear()
    #             # break
    #     if len(lines) != 0:
    #         with open(path + "filtered_dbpedia_03_2021.nt", 'a') as writer:
    #             for l2 in lines:
    #                 writer.write(l2)
    #         lines.clear()


        #     break




if __name__ == "__main__":
    main()
