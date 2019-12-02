def checker():
    lines = None
    with open('data2.txt') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        t = line.rstrip().split()[0]
        for x in range(i + 1 ,len(lines)):
            if t == lines[x].rstrip().split()[0]:
                print i, line.rstrip(), x, lines[x].rstrip()
                return "Dupe Found" 

    return "No Dupe found" 

print checker()
