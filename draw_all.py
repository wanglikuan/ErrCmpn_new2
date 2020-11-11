import os
import matplotlib.pyplot as plt

# Please make all files in this path
# Each file following the naming rules: OpMethod_Model_etc.txt
path = './DGSSimulation/result/'
cat = [] # category, 
models = [] # existing models

for txt in os.listdir(path):
    if txt.endswith('.txt') is True:
        OpMethod = txt.split('_')[0]
        if 'BSPB' in OpMethod or 'DGSVRB' in OpMethod:
            Model = txt.split('_')[1] + " (1 byzantine)"
        else:
            Model = txt.split('_')[1]
        cat.append({'filename': txt, 'OpMethod': OpMethod, 'model': Model})
        if Model not in models:
            models.append(Model)

every = 125

# training loss 
for model in models:
    for result in cat:
        if result['model'] == model:
            f = open(path+result['filename'], 'r')
            x = []
            y = []
            epoch_loss = 0.0
            for line in f.readlines():
                split = line[:-1].split('\t')
                epoch_loss += eval(split[2])
                if eval(split[0]) % every == 0:
                    x.append(eval(split[0]))
                    if eval(split[0]) == 0:
                        y.append(epoch_loss)
                    else:
                        y.append(epoch_loss/every)
                    epoch_loss = 0.0
            plt.plot(x, y, label=result['OpMethod'])
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    plt.title(model)
    plt.legend()
    plt.show()

# sparcification ratio 
for model in models:
    for result in cat:
        if result['model'] == model:
            f = open(path+result['filename'], 'r')
            x = []
            y = []
            epoch_sparcification_ratio = 0.0
            for line in f.readlines():
                split = line[:-1].split('\t')
                epoch_sparcification_ratio += eval(split[3])
                if eval(split[0]) % every == 0:
                    x.append(eval(split[0]))
                    if eval(split[0]) == 0:
                        y.append(epoch_sparcification_ratio)
                    else:
                        y.append(epoch_sparcification_ratio/every)
                    epoch_sparcification_ratio = 0.0
            plt.plot(x, y, label=result['OpMethod'])
    plt.xlabel('iterations')
    plt.ylabel('spacification ratio')
    plt.title(model)
    plt.legend()
    plt.show()