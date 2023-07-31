# dataset subset generation based on the relations
class propertySeperator:
    def __init__(self, filename):
        # Store the filename in the object
        self.filename = filename

        # Initialize empty lists to store the lines
        self.architect_lines = []
        self.artist_lines = []
        self.author_lines = []
        self.commander_lines = []
        self.director_lines = []
        self.musicComposer_lines = []
        self.producer_lines = []
        # false
        self.false_architect_lines = []
        self.false_artist_lines = []
        self.false_author_lines = []
        self.false_commander_lines = []
        self.false_director_lines = []
        self.false_musicComposer_lines = []
        self.false_producer_lines = []

    def process_file(self, pred= False):
        if pred ==False:
            # Open the file for reading
            with open(self.filename, "r") as f:
                # Iterate over the lines in the file
                for line in f:
                    # Split the line on "\t"
                    elements = line.split("\t")

                    # Check if the second element is one of the specified values
                    # and append the line to the corresponding list
                    if elements[4] == "True\n":
                        if elements[1] == "<http://dbpedia.org/ontology/architect>":
                            self.architect_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/artist>":
                            self.artist_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/author>":
                            self.author_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/commander>":
                            self.commander_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/director>":
                            self.director_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/musicComposer>":
                            self.musicComposer_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/producer>":
                            self.producer_lines.append(line)
                        else:
                            print("check")
                    elif elements[4] == "False\n":
                        if elements[1] == "<http://dbpedia.org/ontology/architect>":
                            self.false_architect_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/artist>":
                            self.false_artist_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/author>":
                            self.false_author_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/commander>":
                            self.false_commander_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/director>":
                            self.false_director_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/musicComposer>":
                            self.false_musicComposer_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/producer>":
                            self.false_producer_lines.append(line)
                        else:
                            print("check")

                    else:
                        print("check")
                        break
        else:
            # Open the file for reading
            with open(self.filename, "r") as f:
                # Iterate over the lines in the file
                for line in f:
                    # Split the line on "\t"
                    elements = line.split("\t")
                    if elements[4] == ".\n":
                        if elements[1] == "<http://dbpedia.org/ontology/architect>":
                            self.architect_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/artist>":
                            self.artist_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/author>":
                            self.author_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/commander>":
                            self.commander_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/director>":
                            self.director_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/musicComposer>":
                            self.musicComposer_lines.append(line)
                        elif elements[1] == "<http://dbpedia.org/ontology/producer>":
                            self.producer_lines.append(line)
                        else:
                            print("check")
    def write_lines(self, folder):
        # Open the architect file for writing
        with open(folder+"correct/architect.txt", "w") as f:
            # Write the lines to the file
            for line in self.architect_lines:
                f.write(line)

        # Open the artist file for writing
        with open(folder+"correct/artist.txt", "w") as f:
            # Write the lines to the file
            for line in self.artist_lines:
                f.write(line)

        # Open the commander file for writing
        with open(folder+"correct/commander.txt", "w") as f:
            # Write the lines to the file
            for line in self.commander_lines:
                f.write(line)

        # Open the author file for writing
        with open(folder+"correct/author.txt", "w") as f:
            # Write the lines to the file
            for line in self.author_lines:
                f.write(line)

        # Open the director file for writing
        with open(folder+"correct/director.txt", "w") as f:
            # Write the lines to the file
            for line in self.director_lines:
                f.write(line)

        # Open the musicComposer file for writing
        with open(folder+"correct/musicComposer.txt", "w") as f:
            # Write the lines to the file
            for line in self.musicComposer_lines:
                f.write(line)

        # Open the producer file for writing
        with open(folder+"correct/producer.txt", "w") as f:
            # Write the lines to the file
            for line in self.producer_lines:
                f.write(line)

# Open the architect file for writing
        if len(self.false_architect_lines)>0:
            with open(folder+"false/architect.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_architect_lines:
                    f.write(line)

        # Open the artist file for writing
        if len(self.false_artist_lines)>0:
            with open(folder+"false/artist.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_artist_lines:
                    f.write(line)

        # Open the commander file for writing
        if len(self.false_commander_lines)>0:
            with open(folder+"false/commander.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_commander_lines:
                    f.write(line)

        # Open the author file for writing
        if len(self.false_author_lines)>0:
            with open(folder+"false/author.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_author_lines:
                    f.write(line)

        # Open the director file for writing
        if len(self.false_director_lines) > 0:
            with open(folder+"false/director.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_director_lines:
                    f.write(line)

        # Open the musicComposer file for writing
        if len(self.false_musicComposer_lines) > 0:
            with open(folder+"false/musicComposer.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_musicComposer_lines:
                    f.write(line)

        # Open the producer file for writing
        if len(self.false_producer_lines) > 0:
            with open(folder+"false/producer.txt", "w") as f:
                # Write the lines to the file
                for line in self.false_producer_lines:
                    f.write(line)

# dataset = "../data/dbpedia34k/"
dataset = "../data/dbpedia5/"
folders = ["train/","test/"]
preds = False
hfc_preds = "factcheck_veracity_scores/"

predicates = ["architect/","artist/","commander/","director/","musicComposer/","producer/"]
if preds == True:
    for flder in folders:
        seperator = propertySeperator(dataset + hfc_preds + flder.replace("/", "") + "_pred.txt")
        # seperator = propertySeperator("../data/dbpedia34k/test/test_with_time_final.txt")

        # Process the file
        seperator.process_file(pred=preds)

        # Write the lines to separate files
        seperator.write_lines(dataset + hfc_preds + "properties/"+flder)

else:
    for flder in folders:
        # seperator = propertySeperator(dataset+flder+flder.replace("/","")+".txt")
        seperator = propertySeperator(dataset+flder+flder.replace("/","")+"_with_time_final.txt")
        # seperator = propertySeperator("../data/dbpedia34k/test/test_with_time_final.txt")

        # Process the file
        seperator.process_file()

        # Write the lines to separate files
        seperator.write_lines(dataset+flder + "properties/")
