from tokenizer import SimpleTokenizer
from experiments import encoder_experiment, decoder_experiment
from experiments import load_texts
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run encoder and decoder experiments")
    parser.add_argument("--sanity", help="Run sanity check during testing")
    args = parser.parse_args()
    
    if args.sanity:
        utils = True
    else:
        utils = False

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    print("\nRunning encoder experiment ...")
    encoder_experiment(tokenizer, utils=utils, verbose=False)

    print("\nRunning learned positional encoding decoder experiment ...")
    decoder_experiment(tokenizer, utils=utils, verbose=False, model="base")

    print("\nRunning sinusoidal positional encoding decoder experiment ...")
    decoder_experiment(tokenizer, utils=utils, verbose=False, model="sinusoidal")

    print("\nRunning alibi attention decoder experiment ...")
    decoder_experiment(tokenizer, utils=utils, verbose=False, model="alibi")

    print("\nRunning bigbird attention decoder experiment ...")
    decoder_experiment(tokenizer, utils=utils, verbose=False, model="bigbird")

if __name__ == "__main__":
    main()
