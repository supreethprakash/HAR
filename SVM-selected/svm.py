__author__ = 'raghuveer'

import csv
testResult = []
testResult_1 = []
svm_data = []
svm_test_data = []
svm_output = "svm_file.txt"
svm_test = "svm_test.txt"

with open("processed-data.csv", 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            row = map(float, row)
            testResult.append(row)
csvfile.close()


with open("test-data-pca.csv", 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            row = map(float, row)
            testResult_1.append(row)
csvfile.close()

for i in testResult:
    x = int(i[-1])
    att = []
    count = 1
    att.append(x)
    for j in range(len(i)-1):
        y = count
        z = i[j]
        att.append(y)
        att.append(':')
        att.append(z)
        count += 1
    svm_data.append(att)

for i in testResult_1:
    x = int(i[-1])
    att = []
    count = 1
    att.append(x)
    for j in range(len(i)-1):
        y = count
        z = i[j]
        att.append(y)
        att.append(':')
        att.append(z)
        count += 1
    svm_test_data.append(att)


with open(svm_output, 'wb') as f:
    for l in svm_data:
        l = str(l).replace('[','')
        l = l.replace(']','')
        l = l.replace(",","")
        l = l.replace(" '","")
        l = l.replace("' ","")
        f.writelines("%s\n" %l )
f.close()


with open(svm_test, 'wb') as f:
    for l in svm_test_data:
        l = str(l).replace('[','')
        l = l.replace(']','')
        l = l.replace(",","")
        l = l.replace(" '","")
        l = l.replace("' ","")
        f.writelines("%s\n" %l )
f.close()