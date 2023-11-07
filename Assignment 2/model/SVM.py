from libsvm.svmutil import *


class SVM:

    def __init__(self, c):
        self.c = c
        self.model = None

    def svm_fit_pred(self, train_x, train_y, test_x, test_y):

        """
            -s  svm_type: set type of SVM (-s 0, C-SVC)
            -t  kernel_type: set type of kernel function (-t 0, linear)
            -c  cost: set the parameter C
            -b  probability_estimates: whether to train a SVC or SVR model for probability estimates (-b 0, SVC)
        """

        prob = svm_problem(train_y, train_x)
        param = svm_parameter('-s 0 -t 0 -c {} -b 0'.format(self.c))
        self.model = svm_train(prob, param)
        pred_label, pred_acc, pred_val = svm_predict(test_y, test_x, self.model)

        return pred_label, pred_acc, pred_val
