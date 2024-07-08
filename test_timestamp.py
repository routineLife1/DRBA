def get_timesteps(seq, t, n):
    # assert n >= 0 and 0 <= t <= 1 and len(seq) == 2

    for j in range(n):
        iseq = [seq[i] + (seq[i + 1] - seq[i]) * t for i in range(0, len(seq) - 1)]
        k = 0
        while len(iseq):
            seq.insert(2 * k + 1, iseq.pop(0))
            k += 1

    seq = [eval(f'{s:.4f}') for s in seq]

    return seq[1:-1]

print(get_timesteps([0, 1], 0.01, 1))