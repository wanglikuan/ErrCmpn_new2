import matplotlib.pyplot as plt
import os

print(os.getcwd())

f = open('./result/DGSVRB_LR_16_1.txt', 'r')
x = []
y = []
for line in f.readlines():
    _, loss, _, _, epoch, _, _, _, _ = line[:-1].split('\t')
    x.append(eval(epoch))
    y.append(eval(loss))
plt.plot(x, y)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("ResNet(2 workers)")
# plt.savefig('AlexNet.png')
plt.show()
f.close()