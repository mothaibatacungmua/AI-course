import collections

def first(iterable, default=None):
    "Return the first element of an iterable or the next element of a generator; or default."
    try:
        return iterable[0]
    except IndexError:
        return default
    except TypeError:
        return next(iterable, default)


class PartialExpr:
    """Given 'P |'==>'| Q, first form PartialExpr('==>', P), then combine with Q."""
    def __init__(self, op, lhs): self.op, self.lhs = op, lhs
    def __or__(self, rhs): return Expr(self.op, self.lhs, rhs)
    def __repr__(self): return "PartialExpr('{}', {})".format(self.op, self.lhs)


class Expr(object):
    """A mathematical expression with an operator and 0 or more arguments.
    op is a str like '+' or 'sin'; args are Expression.
    Expr('x') or Symbol('x') creates a symbol (a nullary Expr).
    Expr('-', x) creates a unary; Expr('+', x, 1) creates a binary."""

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args


    #Operator overload
    def __neg__(self):      return Expr('-', self)
    def __pos__(self):      return Expr('+', self)
    def __invert__(self):   return Expr('~', self)
    def __add__(self, rhs): return Expr('+', self, rhs)
    def __sub__(self, rhs): return Expr('-', self, rhs)
    def __mul__(self, rhs): return Expr('*', self, rhs)
    def __pow__(self, rhs): return Expr('**', self, rhs)
    def __mod__(self, rhs): return Expr('%', self, rhs)
    def __and__(self, rhs): return Expr('&', self, rhs)
    def __xor__(self, rhs): return Expr('>>', self, rhs)
    def __rshift__(self, rhs):   return Expr('>>', self, rhs)
    def __lshift__(self, rhs):   return Expr('<<', self, rhs)
    def __truediv__(self, rhs):  return Expr('/', self, rhs)
    def __floordiv__(self, rhs): return Expr('//', self, rhs)
    def __matmul__(self, rhs): return Expr('@', self, rhs)

    def __or__(self, rhs):
        "Allow both P | Q, and P |'==>'|Q."
        if isinstance(rhs, Expression):
            return Expr('|', self, rhs)
        else:
            return PartialExpr(rhs, self)

    # Reverse operator overloads
    def __radd__(self, lhs): return Expr('+', lhs, self)
    def __rsub__(self, lhs): return Expr('-', lhs, self)
    def __rmul__(self, lhs): return Expr('*', lhs, self)
    def __rdiv__(self, lhs): return Expr('/', lhs, self)
    def __rpow__(self, lhs): return Expr('**', lhs, self)
    def __rmod__(self, lhs): return Expr('%', lhs, self)
    def __rand__(self, lhs): return Expr('&', lhs, self)
    def __rxor__(self, lhs): return Expr('^', lhs, self)
    def __ror__(self, lhs): return Expr('|', lhs, self)
    def __rrshift__(self, lhs): return Expr('>>', lhs, self)
    def __rlshift__(self, lhs): return Expr('<<', lhs, self)
    def __rtruediv__(self, lhs): return Expr('/', lhs, self)
    def __rfloordiv__(self, lhs): return Expr('//', lhs, self)
    def __rmatmul__(self, lhs): return Expr('@', lhs, self)

    def __call__(self, *args):
        "Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."
        if self.args:
            raise ValueError('can only do a call for a symbol, not an Expr')
        else:
            return Expr(self.op, args)

    # Equality and repr
    def __eq__(self, other):
        "'x == y' evaluates to True or False; does not build an Expr."
        return (isinstance(other, Expr)
                and self.op == other.op
                and self.args == other.args)

    def __hash__(self): return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():   # f(x) or f(x,y)
            return '{}({})'.format(op, ', '.join(args)) if args else op
        elif len(args == 1):    # -x or -(x + 1)
            return op + args[0]
        else:
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'

Number = (int, float, complex)
Expression = (Expr, Number)


def Symbol(name):
    "A Symbol is just an Expr with no args"
    return Expr(name)


def symbols(names):
    "Return a tuple of Symbols; names is a comma/whitespace delimited str."
    return tuple(Symbol(name) for name in names.replace(',', ' ').split())


def subexpression(x):
    "Yield the subexpression of an Expression (including x itself)"
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpression(arg)



def expr(x):
    """Shortcut to create an Expression. x is a str in which:
    - identifiers are automatically defined as Symbols.
    - ==> is treated as an infix |'==>'|, as are <== and <=>.
    If x is already an Expression, it is returned unchanged. Example:
    >>> expr('P & Q ==> Q')
    ((P & Q) ==> Q)
    """
    if isinstance(x, str):
        return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol))
    else:
        return x

infix_ops = '==> <== <=>'.split()


def expr_handle_infix_ops(x):
    """Given a str, return a new str with ==> replaced by |'==>'|, etc.
    >>> expr_handle_infix_ops('P ==> Q')
    "P |'==>'| Q"
    """
    for op in infix_ops:
        x = x.replace(op, '|' + repr(op) + '|')
    return x

class defaultkeydict(collections.defaultdict):
    """Like defaultdict, but the default_factory is a function of the key.
    >>> d = defaultkeydict(len); d['four']
    4
    """
    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


