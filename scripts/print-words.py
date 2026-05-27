# From a .combined vocab, prints the first 32k words for ContextLM vocabulary

import sys

word_count = 32768

words = []
for line in open(sys.argv[1], "r"):
    if "word=" in line:
        word = line.split("word=")[1].split(",")[0]
        freq = int(line.split(",f=")[1].split(",")[0])
        words.append((word, freq))

words.sort(key=lambda x: -x[1])

for w in words[:word_count]:
    print(w[0])
