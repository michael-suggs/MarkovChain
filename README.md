# Markov Chain Ngram-Based Text Generator


## Libraries

All libraries utilized within this project are listed below.
Further, these can be automatically installed by running the following command.
```shell script
pip3 install -r requirements.txt
```


### Libraries Utilized

None.


## Instructions

All command-line arguments can be described by running the following command.
```shell script
python3 MarkovChain.py --help
```
Note that `python3` should be exchanged to direct to a Python 3.6+ interpreter present on the hosting system.


### Running

1. Ensure all required modules are installed via pip.
2. If text files in the provided directories are not cleaned, this can be done automatically by providing the following additional argument to step 3.
    - `-p` or `--preprocess` will ensure the project header and license text are removed before processing.
3. Call `python3 MarkovChain.py` with the following positional arguments (in order):
    - `AUTH-DIR-1`: Directory of the first author's source material.
    - `AUTH-DIR-2`: Directory of the second author's source material.
    - `PROB-FILE-1`: Path to the file that will store the first author's ngram probabilities.
    - `PROB-FILE-2`: Path to the file that will store the second author's ngram probabilities.
    - `RESULT-FILE`: Path to the file that will store the generated sentences.
    - _Optional:_ `-p` or `--preprocess`


## Code Description
