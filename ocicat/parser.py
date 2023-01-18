from __future__ import annotations
from typing import Tuple, Iterator, cast
from uuid import uuid4, UUID
from dataclasses import dataclass, field
from functools import partial


class Category:
    def __lt__(self, other: Category) -> Left:
        return Left(arg=other, ret=self)  # self < other

    def __gt__(self, other: Category) -> Right:
        return Right(arg=self, ret=other)  # self > other


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
class Right(Category):
    arg: Category
    ret: Category

    def __repr__(self) -> str:
        ret = str(self.ret)
        match self.ret:
            case Left() | Right():
                ret = '(' + ret + ')'
        arg = str(self.arg)
        match self.arg:
            case Left() | Right():
                arg = '(' + arg + ')'
        return arg + ' > ' + ret


@dataclass(frozen=True)
class Left(Category):
    arg: Category
    ret: Category

    def __repr__(self) -> str:
        ret = str(self.ret)
        match self.ret:
            case Left() | Right():
                ret = '(' + ret + ')'
        arg = str(self.arg)
        match self.arg:
            case Left() | Right():
                arg = '(' + arg + ')'
        return ret + ' < ' + arg


Substitutions = list[tuple[Category, Category]]
Equations = list[tuple[Category, Category]]


def left_type_raising(x: Category) -> Tuple[Category, Substitutions]:
    y = Unbound()
    return (y < x) > y, []


def right_type_raising(x: Category) -> Tuple[Category, Substitutions]:
    y = Unbound()
    return y < (x > y), []


def left_composition(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match x, y:
        case Unbound(), Unbound():
            arg, mid, ret = Unbound(), Unbound(), Unbound()
            return ret < arg, [(x, ret < mid), (y, mid < arg)]
        case Unbound(), Left(yarg, yret):
            ret = Unbound()
            return ret < yarg, [(x, ret < yret)]
        case Left(xarg, xret), Unbound():
            arg = Unbound()
            return xret < arg, [(y, xarg < arg)]
        case Left(xarg, xret), Left(yarg, yret):
            subs = unify([(xarg, yret)])
            if subs is not None:
                return xret < yarg, subs
    return None


def right_composition(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match x, y:
        case Unbound(), Unbound():
            arg, mid, ret = Unbound(), Unbound(), Unbound()
            return arg > ret, [(x, arg > mid), (y, mid > ret)]
        case Unbound(), Right(yarg, yret):
            arg = Unbound()
            return arg > yret, [(x, arg > yarg)]
        case Right(xarg, xret), Unbound():
            ret = Unbound()
            return xarg > ret, [(y, xret > ret)]
        case Right(xarg, xret), Right(yarg, yret):
            subs = unify([(xret, yarg)])
            if subs is not None:
                return xarg > yret, subs
    return None


def left_application(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match x:
        case Unbound():
            arg, ret = y, Unbound()
            return ret, [(x, ret < arg)]
        case Left(arg, ret):
            subs = unify([(arg, y)])
            if subs is not None:
                return ret, subs
    return None


def right_application(x: Category, y: Category) \
        -> Tuple[Category, Substitutions] | None:
    match y:
        case Unbound():
            arg, ret = x, Unbound()
            return ret, [(y, arg > ret)]
        case Right(arg, ret):
            subs = unify([(arg, x)])
            if subs is not None:
                return ret, subs
    return None


def substitute(subs: Substitutions, cat: Category) -> Category:
    for (key, value) in subs:
        if cat == key:
            return value
    match cat:
        case Left(arg, ret):
            return substitute(subs, ret) < substitute(subs, arg)
        case Right(arg, ret):
            return substitute(subs, arg) > substitute(subs, ret)
    return cat


def unbounds(cat: Category) -> set[Category]:
    match cat:
        case Unbound():
            return {cat}
        case  Left(arg, ret) | Right(arg, ret):
            return unbounds(arg) | unbounds(ret)
    return set()


def unify(eqs: Equations) -> Substitutions | None:
    match eqs:
        case []:
            return []
        case [(lhs, rhs), *rest] if lhs == rhs:
            return unify(rest)
        case [(Unbound() as lhs, rhs), *rest] if lhs not in unbounds(rhs):
            sub = partial(substitute, [(cast(Category, lhs), rhs)])
            rest = [(sub(lhs), sub(rhs)) for (lhs, rhs) in rest]
            subs = unify(rest)
            return None if subs is None else [*subs, (lhs, rhs)]
        case [(lhs, Unbound() as rhs), *rest] if rhs not in unbounds(lhs):
            sub = partial(substitute, [(cast(Category, rhs), lhs)])
            rest = [(sub(lhs), sub(rhs)) for (lhs, rhs) in rest]
            subs = unify(rest)
            return None if subs is None else [*subs, (rhs, lhs)]
        case [(Right(larg, lret), Right(rarg, rret)), *rest] | \
             [(Left(larg, lret),  Left(rarg, rret)), *rest]:
            return unify([*rest, (larg, rarg), (lret, rret)])
    return None


unary_rules = [
    right_type_raising, left_type_raising,
]

binary_rules = [
    right_composition, left_composition,
    right_application, left_application
]

Tree = Category | tuple['Tree', 'Tree']


def trees(cats: list[Category]) -> Iterator[Tree]:
    if len(cats) == 1:
        yield cats[0]
    else:
        for i in range(1, len(cats)):
            for left in trees(cats[:i]):
                for right in trees(cats[i:]):
                    yield (left, right)


def lift(cat: Category, n: int = -1) -> Iterator[Category]:
    if n == -1:
        i = 0
        while True:
            yield from lift(cat, i)
            i += 1
    elif n == 0:
        yield cat
    else:
        new_cat, _ = right_type_raising(cat)
        yield from lift(new_cat, n - 1)
        new_cat, _ = left_type_raising(cat)
        yield from lift(new_cat, n - 1)


def count(cat: Category) -> int:
    match cat:
        case Right(arg, ret) | Left(arg, ret):
            return count(arg) + count(ret) + 1
    return 0


def dparse1(tree: Tree) -> Category | None:
    match tree:
        case Category():
            return tree
        case (ltree, rtree):
            lcat = dparse1(ltree)
            rcat = dparse1(rtree)
            if lcat is None or rcat is None:
                return None
            for lcatn in lift(lcat):
                if count(lcatn) > count(rcat):
                    break
                for rule in [left_application, right_application]:
                    result = rule(lcatn, rcat)
                    if result is not None:
                       cat, subs = result
                       return substitute(subs, cat)
            for rcatn in lift(rcat):
                if count(rcatn) > count(lcat):
                    break
                for rule in [left_application, right_application]:
                    result = rule(lcat, rcatn)
                    if result is not None:
                       cat, subs = result
                       return substitute(subs, cat)


def dparse(cats: list[Category]) -> Iterator[Category]:
    for tree in trees(cats):
        result = dparse1(tree)
        if result is not None:
            yield result


def parse(cats: list[Category]) -> Iterator[Category]:
    queue = [cats]
    while len(queue) > 0:
        print(len(queue))
        cats = queue.pop(0)
        if len(cats) == 1:
            yield cats[0]
            continue
        for unary_rule in unary_rules:
            for i in range(len(cats)):
                result = unary_rule(*cats[i:i+1])
                if result is None:
                    continue
                cat, subs = result
                new_cats = [*cats[:i], cat, *cats[i+1:]]
                new_cats = [substitute(subs, new_cat) for new_cat in new_cats]
                queue.append(new_cats)
        for binary_rule in binary_rules:
            for i in range(len(cats)-1):
                result = binary_rule(*cats[i:i+2])
                if result is None:
                    continue
                cat, subs = result
                new_cats = [*cats[:i], cat, *cats[i+2:]]
                new_cats = [substitute(subs, new_cat) for new_cat in new_cats]
                queue.append(new_cats)


w, x, y, z = [Atomic(c) for c in 'w,x,y,z'.split(',')]
result = dparse([z < ((w < y) > w), x, x > y])
for c in result:
    print(c)
