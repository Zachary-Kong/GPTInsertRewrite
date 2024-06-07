from collections import Counter
import pandas as pd
import argparse

def create_datalist(path):
    s1_list = []
    s2_list = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('Input'):
                s1_list.append(line[6:])
            elif line.startswith('Paraphrase'):
                s2 = line[line.find('<paraphrase>'):]
                s2_list.append(s2)

    return s1_list, s2_list


def similarity_score(s1_list, s2_list):
    sim_scores = []
    for i in range(len(s1_list)):
        s1 = s1_list[i]
        s2 = s2_list[i]
        s1_i = s1.find('<p1>')
        s1_j = s1.find('<p1/>')

        s2_temp = s2[:s2.find('<p2/>')]
        s2_i = s2_temp.rfind('<p2>')
        s2_j = s2.rfind('<p2/>')

        s1_center = s1[s1_i:s1_j].split()
        s2_center = s2[s2_i:s2_j].split()


        total_words = len(s2_center)
        s1_words = Counter(s1_center)
        s2_words = Counter(s2_center)
        num_new_words = (s2_words-s1_words).total()

        sim_scores.append(num_new_words/total_words)
    return sim_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to data')
    opt = parser.parse_args()
    s1, s2 = create_datalist(opt.path)
    scores = similarity_score(s1, s2)
    avg = sum(scores)/len(scores)
    print('Scores: ', scores)
    print('AVG Score: ', avg)
    