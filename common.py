
def readcat(catPath):
    p = []
    with open(catPath, "r") as f:
        inp = [int(x) for x in f.readline().split()]
        cnt = inp[0]
        assert(cnt == 9)
        for i in range(1, 1 + 2 * 9, 2):
            p.append((inp[i], inp[i + 1]))
    return p
