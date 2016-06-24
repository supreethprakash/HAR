#This command will execute SVM code
import os
#from subprocess import call
#call(["./svm_multiclass_learn -c 5000 svm_file.txt model"])
#call(["./svm_multiclass_classify svm_test.txt model predictions"])
# call(["./svm_multiclass_learn","-c 5000 svm_file.txt model"])

os.system("./svm_multiclass_learn -c 5000 svm_file.txt model")

