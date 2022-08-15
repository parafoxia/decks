import sys

N, m = (int(i) for i in sys.argv[1:3])


def calculate_nodes():
    p1 = ((m + 2) * N) ** 0.5
    p2 = 2 * ((N / (m + 2)) ** 0.5)
    p3 = m * ((N / (m + 2)) ** 0.5)
    return round((p1 + p2) * .75), round(p3 * .75)


h1, h2 = calculate_nodes()
print(f"HL1 = \33[1m{h1}\33[0m\nHL2 = \33[1m{h2}\33[0m")
