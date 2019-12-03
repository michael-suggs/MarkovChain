__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"


import argparse


# Create parser and parse required CL args
parser = argparse.ArgumentParser()
parser.add_argument("AUTH-DIR-1", nargs=1, type=str,
                    help="Directory of texts from author #1")
parser.add_argument("AUTH-DIR-2", nargs=1, type=str,
                    help="Directory of texts from author #2")
parser.add_argument("PROB-FILE-1", nargs=1, type=str,
                    help="Output file for author #1's probability distribution")
parser.add_argument("PROB-FILE-2", nargs=1, type=str,
                    help="Output file for author #2's probability distribution")
parser.add_argument("RESULT-FILE", nargs=1, type=str,
                    help="Output file where sentences are saved")


args=parser.parse_args()
