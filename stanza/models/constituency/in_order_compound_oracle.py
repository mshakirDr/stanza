from enum import Enum, auto

from stanza.models.constituency.dynamic_oracle import DynamicOracle
from stanza.models.constituency.parse_transitions import OpenConstituent, CloseConstituent, CompoundUnary


def accept_wrong_label(gold_transition, pred_transition, gold_sequence, transition_type):
    if not isinstance(gold_transition, transition_type):
        return None
    if not isinstance(pred_transition, transition_type):
        return None
    if gold_transition == pred_transition:
        return None

    return gold_sequence

def accept_wrong_open_label(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Since there is a label on the CloseConstituents, we don't need to do anything here

    In fact, it doesn't even work if we teacher force the Opens.
    Then the Closes learn to deterministically mimic the Opens

    Also, even without the labels, we wouldn't be able to do anything
    even if we wanted given the constraints of the transition scheme
    """
    return accept_wrong_label(gold_transition, pred_transition, gold_sequence, OpenConstituent)

def accept_wrong_close_label(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Well... guess that didn't work out the way we wanted

    Not that we can do anything about it anyway
    """
    return accept_wrong_label(gold_transition, pred_transition, gold_sequence, CloseConstituent)

def accept_wrong_unary_label(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Well... guess that didn't work out the way we wanted

    Not that we can do anything about it anyway
    """
    return accept_wrong_label(gold_transition, pred_transition, gold_sequence, CompoundUnary)


class RepairType(Enum):
    """
    Keep track of which repair is used, if any, on an incorrect transition
    """
    def __new__(cls, fn, correct=False):
        """
        Enumerate values as normal, but also keep a pointer to a function which repairs that kind of error
        """
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value + 1
        obj.fn = fn
        obj.correct = correct
        return obj

    def is_correct(self):
        return self.correct

    WRONG_OPEN_LABEL       = (accept_wrong_open_label,)

    WRONG_CLOSE_LABEL      = (accept_wrong_close_label,)

    WRONG_UNARY_LABEL      = (accept_wrong_unary_label,)

    CORRECT                = (None, True)

    UNKNOWN                = None


class InOrderCompoundOracle(DynamicOracle):
    def __init__(self, root_labels, oracle_level):
        super().__init__(root_labels, oracle_level, RepairType)
