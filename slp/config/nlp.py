from enum import Enum


class SPECIAL_TOKENS(Enum):
    """SPECIAL_TOKENS Special Tokens for NLP applications

    Default special tokens values and indices (compatible with BERT):

        * [PAD]: 0
        * [MASK]: 1
        * [UNK]: 2
        * [BOS]: 3
        * [EOS]: 4
        * [CLS]: 5
        * [SEP]: 6
        * [PAUSE]: 7
    """

    PAD = "[PAD]"
    MASK = "[MASK]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    PAUSE = "[PAUSE]"

    @classmethod
    def has_token(cls, token):
        """has_token check if token exists in SPECIAL_TOKENS

        Args:
            token (str): The special token value (e.g. [PAD]), or name (e.g. PAD)

        Returns:
            bool: True if token exists, False if not
        """
        return any(token in {t.name, t.value} for t in cls)

    @classmethod
    def to_list(cls):
        """to_list Convert SPECIAL_TOKENS to list of tokens

        Returns:
            List[str]: list of token values (e.g. ["[PAD]", "[MASK]"...])
        """
        return list(map(lambda x: x.value, cls))
