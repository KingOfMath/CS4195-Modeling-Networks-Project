# coding=utf-8

def parseData(path):
    file = open(path)
    data = []
    line = file.readline()
    while line:
        data.append(line)
        line = file.readline()
    file.close()
    return data

