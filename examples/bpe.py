from morseg.algorithms.tokenizer import BytePairEncoding

words = [" ".join(list(x)) for x in ["alle", "jahre", "wieder", "alles", "neu", "macht", "der", "mai", "fallen"]]

bpe = BytePairEncoding()

bpe.train(words, iterations=100)

for word in words:
    print(bpe(word))

