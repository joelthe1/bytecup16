import sys
file1 = sys.argv[1]
file2 = sys.argv[2]
f1 = open(file1, 'r').readlines()
f2 = open(file2, 'r').readlines()

diff =0
n = len(f1)
for i in range(n):
    val1 = float(f1[i])
    val2 = float(f2[i])
    diff += abs(val1 - val2)
diff /= n
print "average difference:", diff
