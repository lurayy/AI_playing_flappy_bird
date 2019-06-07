a = []
a.append("kabir")
a.append("hero")

try:
    a[1] = "that"
except:
    a.append("that")
print(a)