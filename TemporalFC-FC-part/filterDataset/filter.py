
import sys


def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)

    path = sys.argv[1:][0]
    # /home/umair/Documents/filteringKG/
    i=0
    lines = []
    with open(path+"filtered_dbpedia_03_2021.nt",'w') as writer:
        print ("test")
    with open(path+"dbpedia_03_2021.nt") as f:
        for line in f:
            # print (line)
            i =  i +1
            if "\"" not in line and "dbpedia.org" in line:
                print (line)
                lines.append(line)
            if i%100000==0:
                with open(path+"filtered_dbpedia_03_2021.nt",'a') as writer:
                    for l2 in lines:
                        writer.write(l2)
                lines.clear()
                # break
        if len(lines) != 0:
            with open(path + "filtered_dbpedia_03_2021.nt", 'a') as writer:
                for l2 in lines:
                    writer.write(l2)
            lines.clear()


        #     break




if __name__ == "__main__":
    main()
