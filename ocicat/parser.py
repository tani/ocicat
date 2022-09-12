from __future__ import annotations
from typing import Tuple, Iterator
from uuid import uuid4, UUID
from dataclasses import dataclass, field


class Category:
    def __lt__(self, other: Category) -> LeftFunctional:
        return LeftFunctional(arg=other,ret=self) # self < other

    def __gt__(self, other: Category) -> RightFunctional:
        return RightFunctional(arg=self, ret=other) # self > other


@dataclass(frozen=True)
class Unbound(Category):
    uuid: UUID = field(default_factory=uuid4)

    def __repr__(self) -> str:
        return '$' + str(self.uuid)[:8]


@dataclass(frozen=True)
class Atomic(Category):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class RightFunctional(Category):
    arg: Category
    ret: Category

    def __repr__(self) -> str:
        ret = str(self.ret)
        match self.ret:
            case LeftFunctional() | RightFunctional():
                ret = '(' + ret + ')'
        arg = str(self.arg)
        match self.arg:
            case LeftFunctional() | RightFunctional():
                ret = '(' + arg + ')'
        return arg + '\\' + ret


@dataclass(frozen=True)
class LeftFunctional(Category):
    arg: Category
    ret: Category

    def __repr__(self) -> str:
        ret = str(self.ret)
        match self.ret:
            case LeftFunctional() | RightFunctional():
                ret = '(' + ret + ')'
        arg = str(self.arg)
        match self.arg:
            case LeftFunctional() | RightFunctional():
                ret = '(' + arg + ')'
        return ret + '//' + arg


Substitutions = list[Tuple[Category, Category]]
Equations = list[Tuple[Category, Category]]


def left_type_raising(x: Category) -> Tuple[Category, Substitutions]:
    y = Unbound()
    return (y < x) > y, []


def right_type_raising(x: Category) -> Tuple[Category, Substitutions]:
    y = Unbound()
    return y < (x > y), []


def left_blue_bird(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match x, y:
        case Unbound(), Unbound():
            arg, mid, ret = Unbound(), Unbound(), Unbound()
            return ret < arg, [(x, ret < mid), (y, mid < arg)]
        case Unbound(), LeftFunctional(yarg, yret):
            ret = Unbound()
            return ret < yarg, [(x, ret < yret)]
        case LeftFunctional(xarg, xret), Unbound():
            arg = Unbound()
            return xret < arg, [(y, xarg < arg)]
        case LeftFunctional(xarg, xret), LeftFunctional(yarg, yret):
            subs = unify([(xarg, yret)])
            if subs is not None:
                return xret < yarg, subs
    return None


def right_blue_bird(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match x, y:
        case Unbound(), Unbound():
            arg, mid, ret = Unbound(), Unbound(), Unbound()
            return arg > ret, [(x, arg > mid), (y, mid > ret)]
        case Unbound(), RightFunctional(yarg, yret):
            arg = Unbound()
            return arg > yret, [(x, arg > yarg)]
        case RightFunctional(xarg, xret), Unbound():
            ret = Unbound()
            return xarg > ret, [(y, xret > ret)]
        case RightFunctional(xarg, xret), RightFunctional(yarg, yret):
            subs = unify([(xret, yarg)])
            if subs is not None:
                return xarg > yret, subs
    return None


def left_identity_bird(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match x:
        case Unbound():
            arg, ret = y, Unbound()
            return ret, [(x, ret < arg)]
        case LeftFunctional(xarg, xret):
            subs = unify([(xarg, y)])
            if subs is not None:
              return xret, subs
    return None


def right_identity_bird(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match y:
        case Unbound():
            arg, ret = x, Unbound()
            return ret, [(y, arg > ret)]
        case RightFunctional(yarg, yret):
            subs = unify([(yarg, x)])
            if subs is not None:
                return yret, subs
    return None


def substitute(cat: Category, subs: Substitutions) -> Category:
    for (key, value) in subs:
        if cat == key:
            return value
    match cat:
        case LeftFunctional(arg, ret):
            return substitute(ret, subs) < substitute(arg, subs)
        case RightFunctional(arg, ret):
            return substitute(arg, subs) > substitute(ret, subs)
    return cat


def unbounds(cat: Category) -> set[Category]:
    match cat:
        case Unbound():
            return {cat}
        case  LeftFunctional(arg, ret) | RightFunctional(arg, ret):
            return unbounds(arg) | unbounds(ret)
    return set()


def unify(eqs: Equations) -> Substitutions | None:
    match eqs:
        case []:
            return []
        case [(lhs, rhs), *rest] if lhs == rhs:
            return unify(rest)
        case [(Unbound() as lhs, rhs), *rest] if lhs not in unbounds(rhs):
            sub = lambda a: substitute(a, [(lhs, rhs)])
            rest = [(sub(lhs), sub(rhs)) for (lhs, rhs) in rest]
            subs = unify(rest)
            return None if subs is None else [*subs, (lhs, rhs)]
        case [(lhs, Unbound() as rhs), *rest] if rhs not in unbounds(lhs):
            sub = lambda a: substitute(a, [(rhs, lhs)])
            rest = [(sub(lhs), sub(rhs)) for (lhs, rhs) in rest]
            subs = unify(rest)
            return None if subs is None else [*subs, (rhs, lhs)]
        case [(RightFunctional(larg, lret), RightFunctional(rarg, rret)), *rest] \
           | [(LeftFunctional(larg, lret),  LeftFunctional(rarg, rret)), *rest]:
            return unify([*rest, (larg, rarg), (lret, rret)])
    return None

unary_rules = [
    right_type_raising, left_type_raising,
]

binary_rules = [
    right_blue_bird, left_blue_bird,
    right_identity_bird, left_identity_bird
]

def parse(cats: list[Category]) -> Iterator[Category]:
    catsq = [cats]
    while len(catsq) > 0:
        cats = catsq.pop(0)
        if len(cats) == 1:
            yield cats[0]
        for unary_rule in unary_rules:
            for i in range(len(cats)-1):
                cat, _ = unary_rule(*cats[i:i+1])
                new_cats = [*cats[:i], cat, *cats[i+1:]]
                catsq = [*catsq, new_cats]
        for binary_rule in binary_rules:
            for i in range(len(cats)-1):
                result = binary_rule(*cats[i:i+2])
                if result is None:
                    continue
                cat, subs = result
                new_cats = [*cats[:i], cat, *cats[i+2:]]
                new_cats = [substitute(new_cat, subs) for new_cat in new_cats]
                catsq = [*catsq, new_cats]


x, y, z = [Atomic(c) for c in 'x,y,z'.split(',')]
result = parse([z < y, x, x > y])
print(result.__next__())
