readfile = open('image2dcharge_3mm.csv', 'r')
labels = []
for line in readfile:
    labels.append(float(line.strip().split(',')[1]))

print min(labels)
