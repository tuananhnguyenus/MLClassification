from MLAlgorithm import *
import random

class SyntheticDataGenerator:
    def __init__(self,numgood,numbad):
        self.numgood=numgood
        self.numbad = numbad
        self.feature1 = 4
        self.feature2 = 5
        self.feature3 = 6
        self.res=[]
    def generate(self):
        res=[]
        for _ in range(self.numgood):
            feature1 = random.randint(self.feature1-1,self.feature1+1)
            feature2 = random.randint(self.feature2-1,self.feature2+1)
            feature3 = 0
            res.append([feature1,feature2,feature3,0])
        for _ in range(self.numbad):
            feature1 = random.randint(self.feature1-1,self.feature1+1)
            feature2 = 0
            feature3 = random.randint(self.feature3-1,self.feature3+1)
            res.append([feature1,feature2,feature3,1])
        return res

class DataAmplifier:
    def __init__(self,num_sample,epsilon,in_filename):
        self.num_sample= num_sample
        self.epsilon = epsilon
        self.in_filename = in_filename
        
        

    def generate(self,out_file,type='train'):
        fi = open(self.in_filename,"r")
        r=[]
        for row in fi:
            for _ in range(self.num_sample):
                t=[]
                vals = row.split(',')
                for val in vals[:-1]:
                    x= random.uniform(1-self.epsilon,1+self.epsilon)
                    tt= round(float(val)*x,2)
                    t.append(str(tt))
                if type=='train': t.append(vals[-1][:-1])
                
                r.append(','.join(t)+'\n')
        fo =open(out_file,"w")
        r=  ''.join(r)
        fo.write(r)
        fo.close()
        return ''.join(r)


synthetic_data = SyntheticDataGenerator(10,20).generate()

synthetic_file = "synthetic_data_100.csv"

np.savetxt(synthetic_file, synthetic_data, delimiter=",", fmt='%s')

trainfile = "train_data.csv"
testfile = "test_data.csv"

generator = DataAmplifier(10,0.05,synthetic_file)
generator.generate(trainfile,'train')
generator.generate(testfile,'test')

random_forest = RandomForestAlg()
random_forest.train(trainfile)
random_forest.predict(testfile)

kmeans= KmeansAlg()
kmeans.train(trainfile)
kmeans.predict(testfile)

decision_tree = DecisionTreeAlg()
decision_tree.train(trainfile)
decision_tree.predict(testfile)

rnn = RNNAlg()
rnn.train(trainfile)
rnn.predict(testfile)

lstm = LSTMAlg()
lstm.train(trainfile)
lstm.predict(testfile)

exit(0)