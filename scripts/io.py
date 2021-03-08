#!/usr/bin/env python3

class FastaReader:

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, "r") as f:
            header = None
            seq = ""
            while True:
                try:
                    line = next(f)
                    if ">" in line:
                        if not header:
                            header = line.strip()
                            seq = ""
                            continue
                        else:
                            rec = (header, seq)
                            header = line.strip()
                            seq = ""
                            yield rec
                    else:
                        seq += line.strip()

                except StopIteration:
                    break


class Kmerize:
    def __init__(self, k=17):
        self.k = k

    def get_kmers(self, s):
        for i in range(len(s) - self.k + 1):
            yield s[i:i+self.k]

    def process(self, iterable):
        for header, seq in iterable:
            for kmers in self.get_kmers(seq):
                yield kmers


def main():
    fa_path = "../data/yeast-upstream-1k-negative.fa"

    fa = FastaReader(fa_path)
    km = Kmerize()

    for kmer in km.process(fa):
        print(kmer)


if __name__ == '__main__':
    main()
