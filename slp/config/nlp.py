from enum import Enum

from typing import List, Type, TypeVar

SpecialTokens = TypeVar('SpecialTokens', bound='SPECIAL_TOKENS')


class SPECIAL_TOKENS(Enum):
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'
    BOS = '[BOS]'
    EOS = '[EOS]'
    CLS = '[CLS]'

    @classmethod
    def has_token(cls: Type[SpecialTokens], token: str) -> bool:
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls: Type[SpecialTokens]) -> List:
        # FIXME: mypy doesn't recognize map can be applied to cls
        return list(map(lambda x: x.value, cls))  # type: ignore
