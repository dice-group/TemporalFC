
import sys

# /home/umair/Documents/pythonProjects/HybridFactChecking/dataset/textResult6.txt
def main():
    data_dir =  '../dataset/'
    output_file = 'textResult7.txt'
    skip= False
    with open(data_dir +output_file, "w") as data_file2:
        print("new file created")
    with open(data_dir+"textResult6.txt" , "r") as file1:
        record = ""
        for line in file1:
            record += line
            if line.__contains__(" predicate team") or line.__contains__("/wrong/date/"):
                skip = True
            if line.startswith("-=-=-=-=-=-=-==-=-==-=-=-=-="):
                with open(data_dir + output_file, "a") as file2:
                    if skip == False:
                        file2.write(record)
                record = ""
                skip = False
            print(line)










if __name__ == "__main__":
    main()
