import torch
from collections import Counter
from typing import List, Dict


class TextPreprocessor:
    def __init__(self, vocab_size: int, max_length: int):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1}
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, corpus: List[Dict[str, any]]):
        """Build vocabulary from a list of sentences (with their source_ids)."""
        tokens = [word for data in corpus for word in data["text"].split()]
        word_freq = Counter(tokens)
        most_common_tokens = word_freq.most_common(
            self.vocab_size - len(self.special_tokens)
        )
        idx = len(self.special_tokens)
        for word, _ in most_common_tokens:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1
        for token, index in self.special_tokens.items():
            self.word2idx[token] = index
            self.idx2word[index] = token

    def tokenize(self, text: str) -> List[int]:
        """Converts text to a list of token ids."""
        tokens = text.split()
        token_ids = [
            self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens
        ]
        return token_ids

    def pad_and_truncate(self, token_ids: List[int]) -> torch.Tensor:
        """Pad or truncate the list of token ids to the max_length."""
        if len(token_ids) > self.max_length:
            return torch.tensor(token_ids[: self.max_length])
        elif len(token_ids) < self.max_length:
            padded_tokens = token_ids + [self.word2idx["<PAD>"]] * (
                self.max_length - len(token_ids)
            )
            return torch.tensor(padded_tokens)
        return torch.tensor(token_ids)

    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create an attention mask where 1 indicates a token and 0 indicates padding."""
        return (token_ids != self.word2idx["<PAD>"]).long()

    def prepare_data(self, data_entry: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """Prepare text data for model input from a dictionary containing source_id and text."""
        token_ids = self.tokenize(data_entry["text"])
        token_ids_padded = self.pad_and_truncate(token_ids)
        attention_mask = self.create_attention_mask(token_ids_padded)
        return {
            "source_id": torch.tensor([data_entry["source_id"]], dtype=torch.int64),
            "input_ids": token_ids_padded,
            "attention_mask": attention_mask,
        }


corpus = [
    {"source_id": 1, "text": "hello world"},
    {"source_id": 2, "text": "machine learning with transformers"},
    {"source_id": 3, "text": "hello there general kenobi"},
]
preprocessor = TextPreprocessor(vocab_size=100, max_length=10)
preprocessor.build_vocab(corpus)
data = preprocessor.prepare_data(
    {"source_id": 1, "text": "hello world machine learning"}
)
print(data)
