from sklearn import metrics
import single_evalution
class evaluation(object):

    def __init__(self, resultfile, rocfile):

        self.testlabel, self.result = self.read_resultfile(resultfile)
        self.sample_number = len(self.testlabel)
        self.rocfile=rocfile

    def read_resultfile(self, resultfile):

        f = open(resultfile, "r")
        text = f.read()
        f.close()

        testlabel = []
        result = []

        for line in text.splitlines():

            line = line.strip()
            values = line.split()
            pro = float(values[0])
            label = int(float(values[1]))

            testlabel.append(label)
            result.append(pro)

        return testlabel, result

    def process(self):

        T = 200
        span = 1.0/T
        opt_T = 0
        opt_Sen = 0
        opt_Spe = 0
        opt_Acc = 0
        opt_MCC = 0
        opt_TP = 0
        opt_FP = 0
        opt_TN = 0
        opt_FN = 0
        list_Sen = []
        list_Fpr = []
        output = open(self.rocfile, 'w')

        for i in range(T+1):
            t=round(i*span,3)
            se = single_evalution.single_evalution(self.testlabel, self.result, i, span)
            se.single_process()
            Sen, Spe, Acc, MCC, TP, FP, TN, FN =se.get_evaluation_index()
            list_Sen.append(Sen)
            list_Fpr.append(1-Spe)
            if(MCC>=opt_MCC):
                opt_T = t
                opt_Sen = Sen;
                opt_Spe = Spe
                opt_Acc = Acc
                opt_MCC = MCC
                opt_TP = TP
                opt_FP = FP
                opt_TN = TN
                opt_FN = FN
            line="T="+str(round(t, 3))+" Sen="+str(round(Sen,3))+" Spe="+str(round(Spe, 3))+" Acc="+str(round(Acc,3))+" MCC="+str(round(MCC,3))+ " TP=" + str(TP) + " FP=" + str(FP) + " TN=" + str(TN) + " FN=" + str(FN) + "\n"
            output.write(line)
        output.flush()
        output.close()
        auc = round(metrics.auc(list_Fpr, list_Sen),3)

        return auc






