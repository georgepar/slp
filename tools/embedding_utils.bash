echo "THIS IS NOT RUNNABLE"; exit 1
#
# Retrieve vectors from model for list of words
#
awk 'FNR==NR{a[$1];next} ($1 in a)' list-of-words /data/models/embeddings/text/fastText/wiki.en.vec > word-vectors

#
# Get vocabulary from word-vectors file
#
cut -d' ' -f1 word-vectors | sort > existing-word-vectors

#
# Compare list-of-words with existing-word-vectors
#
diff existing-word-vectors list-of-words | grep '>' | cut -d' ' -f2 > oov-words

#
# Augment model with OOV words
#
./fasttext print-word-vectors /data/models/embeddings/text/fastText/wiki.en.bin < oov-words >> word-vectors


