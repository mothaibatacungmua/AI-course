#!/usr/bin/env python3

# https://github.com/aimacode/aima-python/blob/master/logic.py
from .utils import (
    Expr, expr, first, subexpression
)
import itertools

class KB:
    def __init__(self, sentence=None):
        raise NotImplementedError

    def tell(self, sentence):
        "Add the sentence to the KB"
        raise NotImplementedError

    def ask(self, query):
        """Return a subsitution that makes the query true, or, failing that, return false"""
        raise NotImplementedError

    def ask_generator(self, query):
        "Yield all the substitutions that make query true"
        raise NotImplementedError

    def retract(self, sentence):
        "Remove sentence from the KB"
        raise NotImplementedError


class PropKB(KB):
    def __init__(self, sentence=None):
        self.clauses = []
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        "Add the sentence's clauses to the KB"
        self.clauses.extend(conjuncts(to_cnf(sentence)))

    def ask_generator(self, query):
        "Yield the empty substitution {} if KB entails query; else no results"
        if tt_entails(Expr('&', self.clauses), query):
            yield {}

    def ask_if_true(self, query):
        "Return True if the KB entails query, else return False."
        for _ in self.ask_generator(query):
            return True
        return False

    def retract(self, sentence):
        "Remove the sentence's clauses from the KB."
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)


def KB_AgentProgram(KB):
    """A generic logical knowledge-based agent program. [Figure 7.1]"""
    steps = itertools.count()

    def program(percept):
        t = next(steps)
        KB.tell(make_percept_sentence(percept, t))
        action = KB.ask(make_action_query(t))
        KB.tell(make_action_sentence(action, t))
        return action

    def make_percept_sentence(self, percept, t):
        return Expr("Percept")(percept, t)

    def make_action_query(self, t):
        return expr("ShouldDo(action, {})".format(t))

    def make_action_sentence(self, action, t):
        return Expr("Did")(action[expr('action')], t)

    return program
#______________________________________________________________________


def is_symbol(s):
    "A string s is a symbol if it starts with an alphabetic char"
    return isinstance(s, str) and s[:1].isalpha()


def is_var_symbol(s):
    "A logic variable symbol is an initial-lowercase string"
    return is_symbol(s) and s[0].islower()

def is_prop_symbol(s):
    "A proposition logic symbol is an initial-uppercase string"
    return is_symbol(s) and s[0].isupper()


def prop_symbols(x):
    "Return a list of all propositional symbols in x."
    if not isinstance(x, Expr):
        return []
    elif is_prop_symbol(x.op):
        return [x]
    else:
        return list(set(symbol for arg in x.args for symbol in prop_symbols(arg)))


def variables(s):
    """Return a set of the variables in expression s.
    >>> variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)')) == {x, y ,z}
    True
    """
    return {x for x in subexpression(s) is is_variable(x)}


#_______________________________________________________________________


def tt_true(s):
    """Is a propositional senetence a tautology?
    >>> tt_true('P | ~P')
    True
    """
    s = expr(s)
    return tt_entails(True, s)




def tt_entails(kb, alpha):
    """Does kb entail the sentence alpha? Use truth tables. For propositional
    kb's and sentences. [Figure 7.10]. Note that the 'kb' should be an Expr
    which is a conjunction of clauses.
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    assert not variables(alpha)
    return tt_check_all(kb, alpha, prop_symbols(kb & alpha), {})


def tt_check_all(kb, alpha, symbols, model):
    "Auxiliary routine to implement tt_entails"
    if not symbols:
        if pl_true(kb, model):
            result = pl_true(alpha, model)
            assert result in (True, False)
            return result
        else:
            return True

def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for every
    proposition, this may return None to indicate 'not obvious';this may happen
    even when the expression is tautological.
    """
    if exp in (True, False):
        return exp

    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                return None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                return None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^': #xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError('illegal operator in logic expression' + str(exp))



# Convert to Conjunctive Normal Form (CNF)
def to_cnf(s):
    """Convert a propositional logical sentence to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...) [p.253]
    >>> to_cnf('~(B | C)')
    (~B & ~C)
    """
    s = expr(s)
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implication(s)
    s = move_not_inwards(s)
    return distribute_and_over_or(s)

def eliminate_implication(s):
    "Change implication into equivalent form with only &, |, and ~ as logical operators."
    s = expr(s)
    if not s.args or is_symbol(s.op):
        return s
    args = list(map(eliminate_implication, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2 #TODO: relax this restriction
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    """
    s = expr(s)
    if s.op == '~':
        def NOT(b):
            return move_not_inwards(~b)
        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s

def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    s = expr(s)
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return False
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = first(arg for arg in s.args if arg.op == '&')
        if not conj:
            return s

        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest) for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


_op_identity = {'&': True, '|': False, '+': 0, '*': 1}


def associate(op, args):
    """Given an associate op, return an expression with the same
    meaning as Expr(op, *args), but flattened -- that is, with nested
    instances of the same op promoted to the top level
    >>> associate('&', [(A&B),(B|C), (B&C)])
    (A & B & (B|C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)


def dissociate(op, args):
    """Given an associative op, return a flattened list result such
    that Expr(op, *result) means the same as Expr(op, *args).
    """
    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result


def conjuncts(s):
    """Return a list of the conjuncts in the sentence s
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])


def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])


def extend(s, var, val):
    "Copy the substitution s and extend it by setting var to val; return copy"
    s2 = s.copy()
    s2[var] = val
    return s2

def is_variable(x):
    "A variable is an Expr with no args and a lowercase symbol as the op."
    return isinstance(x, Expr) and not x.args and x.op[0].islower()