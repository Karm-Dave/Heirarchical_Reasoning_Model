def neumann_series(jvp_fn, v, k):
    out = v
    acc = v
    for _ in range(k):
        out = jvp_fn(out)
        acc = acc + out
    return acc