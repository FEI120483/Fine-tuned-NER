"""
    将原始数据转换为BME格式
"""

import codecs

input_data_path = 'raw_data/testright.txt'
output_data_path = 'ner_data/test'
def Transform_Format(input_data_path, output_data_path):
    input = codecs.open(input_data_path, 'r', encoding='gbk')
    output = codecs.open(output_data_path, 'w', encoding='utf-8')
    for line in input:
        line = line.strip().split()
        if len(line) == 0:
            continue

        for word in line:
            word = word.split('/')
            
            if word[1] != 'o':
                if len(word[0]) == 1:
                    output.write(word[0] + ' B-' + word[1] + '\n')
                elif len(word[0]) == 2:
                    output.write(word[0][0] + ' B-' + word[1] + '\n')
                    output.write(word[0][1] + ' E-' + word[1] + '\n')
                else:
                    output.write(word[0][0] + ' B-' + word[1] + '\n')
                    for i in word[0][1: len(word[0])-1]:
                        output.write(i + ' M-' + word[1] + '\n')
                    output.write(word[0][-1] + ' E-' + word[1] + '\n')
            else:
                for i in word[0]:
                    output.write(i + ' o' + '\n')
        output.write('\n')
    input.close()
    output.close()

if __name__ == '__main__':
    Transform_Format(input_data_path, output_data_path)
    with open(output_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            print(line)
            break