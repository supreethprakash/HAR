#This code is written by Supreeth for the purpose of merging train test with subject data.

import csv
import os
import operator

dataset = []
dataset1 = []
data_set_for = []

ytrain ="y_train.txt"
ytest ="y_test.txt"
subjectTrain = "subject_train.txt"
subjectTest = "subject_test.txt"
ground_truth = "ground_truth.csv"

#-----------------------------------------------------
#File Names for generated output
#-----------------------------------------------------
pca_knn_sub = "pca_knn_sub.csv"
pca_nb_sub = "pca_nb_sub.csv"
pca_dt_sub = "pca_dt_sub.csv"
cor_knn_sub = "cor_knn_sub.csv"
cor_nb_sub = "cor_nb_sub.csv"
cor_dt_sub = "cor_dt_sub.csv"
var_knn_sub = "var_knn_sub.csv"
var_nb_sub = "var_nb_sub.csv"
var_dt_sub = "var_dt_sub.csv"
wr_knn_sub = "wr_knn_sub.csv"
wr_nb_sub = "wr_nb_sub.csv"
wr_dt_sub = "wr_dt_sub.csv"

#------------------------------------------------------
naive_cor_predicted = "naive_cor_predicted.txt"
naive_noreduction_predicted = "naive_noreduction_predicted.txt"
naive_pca_predicted = "naive_pca_predicted.txt"
naive_var_predicted = "naive_var_predicted.txt"
#------------------------------------------------------
prediction_nored = "prediction_nored.csv"
predictionCor = "predictionCor.csv"
predictionPCA = "predictionPCA.csv"
predictionVar = "predictionVar.csv"
#------------------------------------------------------
pca_predicted_dt = "pca_predicted_dt.txt"
cor_predicted_dt = "cor_predicted_dt.txt"
var_predicted_dt= "var_predicted_dt.txt"
noreduction_predicted_dt = "noreduction_predicted_dt.txt"
#------------------------------------------------------

def openytrain():
    with open(ytrain,"r") as f:
        reader = csv.reader(f, dialect='excel', delimiter=' ')
        for row in reader:
            item = int(row[0])
            dataset.append(int(item))
    f.close()

def openytrain_fordiverse():
    with open(ytrain,"r") as f:
        reader = csv.reader(f, dialect='excel', delimiter=' ')
        for row in reader:
            item = int(row[0])
            data_set_for.append(int(item))
    f.close()

def openytest():
    with open(ytest,"r") as f:
        reader = csv.reader(f, dialect='excel', delimiter=' ')
        for row in reader:
            item = int(row[0])
            dataset.append(int(item))
    f.close()

def openytest_diverse(reduction_technique, classification_technique):

    if(reduction_technique == 1):
        if(classification_technique == 1):
            with open(predictionPCA,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = float(row[1])
                    data_set_for.append(int(item))
            f.close()

        elif(classification_technique == 2):
            with open(naive_pca_predicted,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()

        elif(classification_technique == 3):
            with open(pca_predicted_dt,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()

    elif(reduction_technique == 2):
        if(classification_technique == 1):
            with open(predictionCor,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = float(row[1])
                    data_set_for.append(int(item))
            f.close()
        elif(classification_technique == 2):
            with open(naive_cor_predicted,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()
        elif(classification_technique == 3):
            with open(cor_predicted_dt,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()

    elif(reduction_technique == 3):
        if(classification_technique == 1):
            with open(predictionVar,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = float(row[1])
                    data_set_for.append(int(item))
            f.close()
        elif(classification_technique == 2):
            with open(naive_var_predicted,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()
        elif(classification_technique == 3):
            with open(var_predicted_dt,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()

    else:
        if(classification_technique == 1):
            with open(prediction_nored,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = float(row[1])
                    data_set_for.append(int(item))
            f.close()
        elif(classification_technique == 2):
            with open(naive_noreduction_predicted,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()
        elif(classification_technique == 3):
            with open(noreduction_predicted_dt,"r") as f:
                reader = csv.reader(f, dialect='excel', delimiter=' ')
                for row in reader:
                    item = int(row[0])
                    data_set_for.append(int(item))
            f.close()

def open_subj_train():
    with open(subjectTrain,"r") as f:
        reader = csv.reader(f, dialect='excel', delimiter=' ')
        for row in reader:
            item = int(row[0])
            dataset1.append(int(item))
    f.close()

def open_subj_test():
    with open(subjectTest,"r") as f:
        reader = csv.reader(f, dialect='excel', delimiter=' ')
        for row in reader:
            item = int(row[0])
            dataset1.append(int(item))
    f.close()

def write_to_file(itemset, opt, selection1, selection):
    if(opt == 0):
        if os.path.isfile(ground_truth) != True:
            with open(ground_truth, 'wb') as f:
                for l in itemset:
                    l = str(l).replace('[','')
                    l = l.replace(']','')
                    f.writelines("%s\n" %l )
            f.close()
    else:
        if selection1 == 1 and selection == 1:
            if os.path.isfile(pca_knn_sub) != True:
                with open(pca_knn_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 1 and selection == 2:
            if os.path.isfile(pca_nb_sub) != True:
                with open(pca_nb_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 1 and selection == 3:
            if os.path.isfile(pca_dt_sub) != True:
                with open(pca_dt_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()

        elif selection1 == 2 and selection == 1:
            if os.path.isfile(cor_knn_sub) != True:
                with open(cor_knn_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 2 and selection == 2:
            if os.path.isfile(cor_nb_sub) != True:
                with open(cor_nb_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 2 and selection == 3:
            if os.path.isfile(cor_dt_sub) != True:
                with open(cor_dt_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 3 and selection == 1:
            if os.path.isfile(var_knn_sub) != True:
                with open(var_knn_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 3 and selection == 2:
            if os.path.isfile(var_nb_sub) != True:
                with open(var_nb_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 3 and selection == 3:
            if os.path.isfile(var_dt_sub) != True:
                with open(var_dt_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 4 and selection == 1:
            if os.path.isfile(wr_knn_sub) != True:
                with open(wr_knn_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 4 and selection == 2:
            if os.path.isfile(wr_nb_sub) != True:
                with open(wr_nb_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()
        elif selection1 == 4 and selection == 3:
            if os.path.isfile(wr_dt_sub) != True:
                with open(wr_dt_sub, 'wb') as f:
                    for l in itemset:
                        l = str(l).replace('[','')
                        l = l.replace(']','')
                        f.writelines("%s\n" %l )
                f.close()

def create_array(opt,selection1,selection):
    if opt == 0:
        itemset = [[0 for i in range(2)] for j in range(len(dataset))]
        for i in range(len(dataset)):
            j = 0
            itemset[i][j] = dataset1[i]
            itemset[i][j+1] = dataset[i]
        write_to_file(itemset, opt, selection1, selection)

    elif opt == 1:
        itemset = [[0 for i in range(2)] for j in range(len(data_set_for))]
        for i in range(len(data_set_for)):
            j = 0
            itemset[i][j] = dataset1[i]
            itemset[i][j+1] = dataset[i]
        write_to_file(itemset, opt, selection1, selection)



def main():
    selection1 = raw_input("Enter the which type of reducton technique? \n 1. PCA 2. Correlation 3. Variance Filter 4. Without Reduction")
    selection = raw_input("Enter the which type classification you want to merge? \n 1. KNN 2. Naive-Bayes 3. Decision-Tree")
    openytrain()
    openytest()
    open_subj_train()
    open_subj_test()
    create_array(0,int(selection1),int(selection))
    openytrain_fordiverse()
    openytest_diverse(int(selection1),int(selection))
    create_array(1,int(selection1),int(selection))


main()