import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # unigram
                for i in range(self.unique_words):
                    idx,token,count=f.readline().strip().split(' ')
                    idx=int(idx)
                    count=int(count)
                    self.index[token]=idx
                    self.word[idx]=token
                    self.unigram_count[idx]=count
                # bigram
                for line in f:
                    if line.strip()=="-1":
                        break
                    idx1,idx2,logp=line.strip().split(' ')
                    idx1=int(idx1)
                    idx2=int(idx2)
                    logp=float(logp)
                    self.bigram_prob[idx1][idx2]=logp
                    self.logProb+=logp
                self.logProb=self.logProb/len(self.bigram_prob)

                
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 

        curr_idx=self.index[w]
        G=[f"{w}"]
        for i in range(n):
            if curr_idx in self.bigram_prob:
                I=list(self.bigram_prob[curr_idx].keys())
                LogP=list(self.bigram_prob[curr_idx].values())
                P= [math.exp(lp) for lp in LogP]
                next_idx=random.choices(I,weights=P,k=1)[0]
                
            else:
                I=list(self.word.keys())
                next_idx=random.choice(I)

            G.append(f" {self.word[next_idx]}")
            curr_idx=next_idx
        print(G)
        return G

            
        


        


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
