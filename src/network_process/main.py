# coding=utf-8
import util.read_file as rf

metadata_path = "../../sources/rel.rating"
UserNodeNum = 943
MovieNodeNum = 1682
EdgeNum = 100000

if __name__ == '__main__':
    data = rf.parseData(metadata_path)
    print(data[0])