__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"


import argparse
from os import listdir, path
from ngram import NGram


# Create parser and parse required CL args
parser = argparse.ArgumentParser()
parser.add_argument("AUTH_DIR_1", nargs=1, type=str,
                    help="Directory of texts from author #1")
# parser.add_argument("AUTH_DIR_2", nargs=1, type=str,
#                     help="Directory of texts from author #2")
parser.add_argument("PROB_FILE_1", nargs=1, type=str,
                    help="Output file for author #1's probability distribution")
# parser.add_argument("PROB_FILE_2", nargs=1, type=str,
#                     help="Output file for author #2's probability distribution")
parser.add_argument("RESULT_FILE", nargs=1, type=str,
                    help="Output file where sentences are saved")


args=parser.parse_args()


def read_file(rfile):
    text = []
    for file in listdir(rfile):
        with open(path.join(rfile, file), 'r') as f:
            book = [line for line in f]
            text.append(book)

    return text


if __name__ == '__main__':
    ng = NGram(text=read_file(args.AUTH_DIR_1[0]))
    ng.parse_ngrams(3)
    ng.generate(20, 10)
    ng.write_results(args.RESULT_FILE[0])
    ng.write_ngrams(args.PROB_FILE_1[0])