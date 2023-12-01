from model import Word2Vec
from tokenizer import Tokenizer
from trainer import Trainer

training_data = ". ".join(
    [
        "cats rule the world",
        "dogs are the best",
        "elephants have long trunks",
        "monkeys like bananas",
        "pandas eat bamboo",
        "tigers are dangerous",
        "zebras have stripes",
        "lions are the kings of the savannah",
        "giraffes have long necks",
        "hippos are big and scary",
        "rhinos have horns",
        "penguins live in the arctic",
        "polar bears are white",
    ]
)

token_indices = Tokenizer().tokenize(training_data)
print(token_indices[:4])

MAX_SEQ_LEN = 5


def create_seq(token_indices: list[int]):
    sequences = []
    for i in range(0, len(token_indices) - MAX_SEQ_LEN + 1):
        sequences.append(token_indices[i : i + MAX_SEQ_LEN])
    return sequences


train_seq = create_seq(token_indices)
print(train_seq[:10])


vocab_size = Tokenizer().size()
EMBEDDING_DIM = 10

model = Word2Vec(vocab_size, EMBEDDING_DIM, Tokenizer().character_to_token("<pad>"))

Trainer(num_epochs=100, batch_size=8).fit(model, train_seq)
