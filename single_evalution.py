import math
class single_evalution():

    def __init__(self, testlabel, result, index, span):

        self.testlabel = testlabel
        self.sample_number= len(self.testlabel)
        self.result = result
        self.t = round(index*span, 3)

    def calculate_1(self, prelabel):

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(self.sample_number):

            if (self.testlabel[i] == 1 and prelabel[i] == 1):
                TP = TP + 1
            if (self.testlabel[i] == 0 and prelabel[i] == 0):
                TN = TN + 1
            if (self.testlabel[i] == 0 and prelabel[i] == 1):
                FP = FP + 1
            if (self.testlabel[i] == 1 and prelabel[i] == 0):
                FN = FN + 1
        return TP, TN, FP, FN

    def calculate2(self, TP, TN, FP, FN):

        Sen=float(TP)/(TP+FN)
        Spe=float(TN)/(TN+FP)
        Acc=float(TP+TN)/(TP+FP+TN+FN)
        MCC=0
        if((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN)!=0):
            MCC=float(TP*TN-FN*FP)/math.sqrt(float(TP+FN)*(TN+FP)*(TP+FP)*(TN+FN))
        return Sen, Spe, Acc, MCC, TP, FP, TN, FN

    def single_process(self):

        prelabel = self.get_prelabel(self.t)
        TP, TN, FP, FN = self.calculate_1(prelabel)
        self.Sen, self.Spe, self.Acc, self.MCC, self.TP, self.FP, self.TN, self.FN = self.calculate2(TP, TN, FP, FN)

    def get_evaluation_index(self):

        return self.Sen, self.Spe, self.Acc, self.MCC, self.TP, self.FP, self.TN, self.FN

    def get_prelabel(self, T):

        pre_label=[]
        for i in range(self.sample_number):
            pre_label.append(self.judge(self.result[i], T))
        return pre_label

    def judge(self, predict_value, T):
        if (predict_value >= T):
            return 1
        else:
            return 0

