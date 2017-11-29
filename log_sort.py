import matplotlib.pyplot as plt
import re

gen = []
avrg = []
medi = []
mini = []
maxi = []
nliv = []
tliv = []

def sort():
    file = open("log.txt", 'r')
    lines = file.readlines()

    for line in lines:
        if 'Average' in line:
            avrg.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))
        elif 'generation' in line:
            gen.append(int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))
        elif 'Median' in line:
            medi.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))
        elif 'Minimum' in line:
            mini.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))
        elif 'Maximum' in line:
            maxi.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))
        elif 'Number lived' in line:
            nliv.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))
        elif 'Total time lived' in line:
            tliv.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]))

    plt.plot(gen, avrg)
    plt.plot(gen, medi)
    # plt.plot(gen, maxi)

if __name__ == "__main__":
    sort()
    print(avrg)