import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Decoder, Encoder
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32# Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
dropout = 0.2
vocab_size = 5755


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def cls_collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Add a special token for the classifier
    data_with_cls = [torch.cat([torch.tensor([vocab_size]), d]) for d in data]
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
            padded_sequences,
            (0, max(0, block_size - padded_sequences.shape[1])), 
            "constant", 
            0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    size = len(data_loader.dataset)
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            preds, _, _ = classifier(X)
            total_correct += (preds.argmax(1) == Y).type(torch.float).sum().item()
        accuracy = total_correct / size
        classifier.train()
        return accuracy

def train_classifier_epoch(classifier, data_loader, optimizer):
    """ Train the classifier on the data in data_loader for the specified number of epochs."""
    classifier.train()
    train_loss, correct = 0, 0
    for batch, (X, Y) in enumerate(data_loader):
        X, Y = X.to(device), Y.to(device)

        preds, loss, _ = classifier(X, Y)
        train_loss += loss.item()
        correct += (preds.argmax(1) == Y).type(torch.float).sum().item() # From hw1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = train_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    return mean_loss, accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=200):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for batch, (X, Y) in enumerate(data_loader):
        X, Y = X.to(device), Y.to(device)
        _, loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def train_decoder_epoch(decoder, data_loader, optimizer, tokenizer):
    """ Train the decoder on the data in data_loader and return mean_loss"""
    decoder.train()
    losses = []
    train_loss = 0
    for batch, (X, Y) in enumerate(data_loader):
        if batch > max_iters:
            break
        X, Y = X.to(device), Y.to(device)

        _, loss, _ = decoder(X, Y)
        losses.append(loss.item())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % eval_interval == 0:
            print(f"\n")
            eval_losses = torch.tensor(losses)
            mean_loss = eval_losses.mean()
            perplexity = torch.exp(mean_loss).item()
            print(f"Batch {batch}: Train loss: {mean_loss}, Train perplexity: {perplexity}")
            for president in ["hbush", "wbush", "obama"]:
                test_dataset = LanguageModelingDataset(tokenizer, read_file(f"speechesdataset/test_LM_{president}.txt"), block_size)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
                test_perplexity = compute_perplexity(decoder, test_loader, eval_iters)
                print(f"Batch {batch}: Test perplexity for {president}: {test_perplexity}")

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()
    mean_loss = train_loss / len(data_loader)
    return mean_loss, perplexity

def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

def encoder_experiment(tokenizer:SimpleTokenizer, utils=False):
    # Encoder
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=cls_collate_batch, shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=cls_collate_batch, shuffle=False)


    classifier = Encoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            block_size=block_size,
            hidden_dim=n_embd * 4, # 4x embed based off "Attention is All You Need" paper
            num_heads=n_head,
            num_layers=n_layer,
            dropout=dropout,
            ).to(device)
    sanity_string = "The third source of tension is our shared interest in the rights and responsibilities of nations on nuclear weapons."
    classifier_utils = Utilities(tokenizer, classifier)
    # print(classifier)
    # if True:
    #     classifier_utils.sanity_check(sanity_string, block_size, device)


    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    # for the classification  task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        train_loss, train_accuracy = train_classifier_epoch(classifier, train_CLS_loader, optimizer)
        test_accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
        print(f"Epoch {epoch}: Train loss: {train_loss}, Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")
    print(f"Number of parameters in the classifier: {sum(p.numel() for p in classifier.parameters())}")
    if utils:
        classifier_utils.sanity_check(sanity_string, block_size, device)

def decoder_experiment(tokenizer, utils=False):
    # Decoder
    train_LM_dataset = LanguageModelingDataset(tokenizer, read_file("speechesdataset/train_LM.txt"),  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    decoder = Decoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            block_size=block_size,
            num_heads=n_head,
            hidden_dim=n_hidden,
            num_layers=n_layer,
            attention="alibi",
            dropout=0.0,
            ).to(device)
    print("Training decoder model ...")

    decoder_utils = Utilities(tokenizer, decoder)
    sanity_string = "The third source of tension is our shared interest in the rights and responsibilities of nations on nuclear weapons."
    if utils:
        decoder_utils.sanity_check(sanity_string, block_size, device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(1):
        train_loss, train_perplexity = train_decoder_epoch(decoder, train_LM_loader, optimizer, tokenizer)
        # print(f"Epoch {epoch}: Train loss: {train_loss}, Train perplexity: {train_perplexity}")
        # if epoch % eval_interval == 0:
        #     print(f"Epoch {epoch}: Train loss: {train_loss}, Train perplexity: {train_perplexity}")
    print(f"Number of parameters in the decoder: {sum(p.numel() for p in decoder.parameters())}")
    if utils:
        decoder_utils.sanity_check(sanity_string, block_size, device)



def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    encoder_experiment(tokenizer, utils=False)
    # decoder_experiment(tokenizer, utils=False)

if __name__ == "__main__":
    main()
