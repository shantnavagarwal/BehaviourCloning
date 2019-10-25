import numpy as np
import matplotlib.pyplot as plt

xtrainorig = np.load('UXTrain1.npy')
ytrainorig = np.load('UYTrain1.npy')

classes = np.unique(ytrainorig)
num, bins = np.histogram(ytrainorig, len(classes))
centre = (bins[1:] + bins[:-1])/2
plt.bar(centre, num, align='center', width=.1)
plt.show()

