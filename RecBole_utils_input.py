from enum import Enum

class InputTpe(Enum):
    """
    Type of Models' input.
    -``POINTWISE``: Point-wise input, like `` uid, iid, label``.
    -``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """
    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3