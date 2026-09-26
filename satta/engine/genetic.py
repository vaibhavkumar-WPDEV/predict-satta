"""Genetic programming: the engine invents its own formulas.

A formula is an expression tree over the values known before a draw
(previous draws, other markets' results of the previous day, their reverses,
the date). Nodes combine them with + − × (mod 100), andar/bahar (digits),
ulta (reverse), cut (+5 per digit), jod (digit sum), beejank (digital root)
and jodi(a, b) (glue two digits into a jodi). The final offset k is fitted as
the most common residual, like the linear search.

A population of random formulas evolves by tournament selection, subtree
crossover and mutation; fitness is how often the formula hit on the training
draws, minus a small size penalty so simple formulas win ties. Evolution is
seeded, so the same data always evolves the same formulas.
"""

from __future__ import annotations

import numpy as np

from .base import cut, rev


def _dr(x):
    return np.where(x == 0, 0, 1 + (x - 1) % 9)


OPS1 = {
    "andar": lambda a: a // 10,
    "bahar": lambda a: a % 10,
    "ulta": rev,
    "cut": cut,
    "jod": lambda a: (a // 10 + a % 10) % 10,
    "beejank": _dr,
}
OPS2 = {
    "+": lambda a, b: (a + b) % 100,
    "−": lambda a, b: (a - b) % 100,
    "×": lambda a, b: (a * b) % 100,
    "jodi": lambda a, b: (a % 10) * 10 + b % 10,
}
MAX_DEPTH = 4


def evaluate(tree, V: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Values of the tree for every row; -1 where an input is missing."""
    kind = tree[0]
    if kind == "var":
        return V[tree[1]]
    if kind == "const":
        return np.full(n, tree[1])
    if kind in OPS1:
        a = evaluate(tree[1], V, n)
        return np.where(a >= 0, OPS1[kind](np.where(a >= 0, a, 0)), -1)
    a = evaluate(tree[1], V, n)
    b = evaluate(tree[2], V, n)
    ok = (a >= 0) & (b >= 0)
    return np.where(ok, OPS2[kind](np.where(ok, a, 0), np.where(ok, b, 0)), -1)


def size(tree) -> int:
    return 1 + sum(size(c) for c in tree[1:] if isinstance(c, tuple))


def to_str(tree, names: dict) -> str:
    kind = tree[0]
    if kind == "var":
        return names.get(tree[1], tree[1])
    if kind == "const":
        return f"{tree[1]}"
    if kind in OPS1:
        return f"{kind}({to_str(tree[1], names)})"
    if kind == "jodi":
        return f"jodi({to_str(tree[1], names)}, {to_str(tree[2], names)})"
    return f"({to_str(tree[1], names)} {kind} {to_str(tree[2], names)})"


def _random_tree(rng, vars_, depth):
    if depth <= 1 or rng.random() < 0.3:
        if rng.random() < 0.85:
            return ("var", vars_[rng.integers(len(vars_))])
        return ("const", int(rng.integers(1, 100)))
    if rng.random() < 0.4:
        op = list(OPS1)[rng.integers(len(OPS1))]
        return (op, _random_tree(rng, vars_, depth - 1))
    op = list(OPS2)[rng.integers(len(OPS2))]
    return (op, _random_tree(rng, vars_, depth - 1), _random_tree(rng, vars_, depth - 1))


def _paths(tree, path=()):
    yield path
    for i, c in enumerate(tree[1:], start=1):
        if isinstance(c, tuple):
            yield from _paths(c, path + (i,))


def _get(tree, path):
    for i in path:
        tree = tree[i]
    return tree


def _put(tree, path, sub):
    if not path:
        return sub
    i = path[0]
    return tree[:i] + (_put(tree[i], path[1:], sub),) + tree[i + 1:]


def _depth(tree) -> int:
    kids = [c for c in tree[1:] if isinstance(c, tuple)]
    return 1 + (max(_depth(c) for c in kids) if kids else 0)


def fit(tree, V, Y) -> dict | None:
    """Fitted offset k, hits and valid rows of `tree + k (mod 100)` on (V, Y)."""
    f = evaluate(tree, V, len(Y))
    ok = f >= 0
    n = int(ok.sum())
    if n < max(20, len(Y) // 2):
        return None
    cnt = np.bincount((Y[ok] - f[ok]) % 100, minlength=100)
    k = int(cnt.argmax())
    return {"tree": tree, "k": k, "hits": int(cnt[k]), "n": n}


def predict(f: dict, V: dict) -> int:
    v = int(evaluate(f["tree"], V, 1)[0])
    return (v + f["k"]) % 100 if v >= 0 else -1


def evolve(V: dict[str, np.ndarray], Y: np.ndarray, seed: int, pop: int = 120,
           gens: int = 25, top: int = 5) -> list[dict]:
    """Evolve formulas on (V, Y). Returns the best distinct formulas, best first."""
    if len(Y) < 30 or not V:
        return []
    rng = np.random.default_rng(seed)
    vars_ = sorted(V)

    def score(f):
        return -1.0 if f is None else f["hits"] / f["n"] - 0.002 * size(f["tree"])

    popn = []
    tries = 0
    while len(popn) < pop and tries < pop * 20:
        tries += 1
        f = fit(_random_tree(rng, vars_, MAX_DEPTH), V, Y)
        if f is not None:
            popn.append(f)
    if not popn:
        return []
    for _ in range(gens):
        popn.sort(key=score, reverse=True)
        nxt = popn[:5]  # elitism
        tries = 0
        while len(nxt) < pop and tries < pop * 20:
            tries += 1
            a = max((popn[i] for i in rng.integers(0, len(popn), 4)), key=score)["tree"]
            r = rng.random()
            if r < 0.6:
                b = max((popn[i] for i in rng.integers(0, len(popn), 4)), key=score)["tree"]
                pa = list(_paths(a))
                pb = list(_paths(b))
                child = _put(a, pa[rng.integers(len(pa))], _get(b, pb[rng.integers(len(pb))]))
            elif r < 0.9:
                pa = list(_paths(a))
                child = _put(a, pa[rng.integers(len(pa))], _random_tree(rng, vars_, 3))
            else:
                child = a
            if _depth(child) > MAX_DEPTH + 1:
                continue
            f = fit(child, V, Y)
            if f is not None:
                nxt.append(f)
        popn = nxt
    popn.sort(key=score, reverse=True)
    out, seen = [], set()
    for f in popn:
        key = repr(f["tree"]) + str(f["k"])
        if key not in seen:
            seen.add(key)
            out.append(f)
        if len(out) == top:
            break
    return out
