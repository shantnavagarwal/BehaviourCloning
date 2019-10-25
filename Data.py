import csv
from scipy import ndimage
import numpy as np

lines = []  # get list containing location of all images.
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

origimages = []
origmeasurements = []
correction = 0.2  # correction fcactor for left/ right camera images

for line in lines:
    sourcepath = line[0]  # for centre images
    filename = sourcepath.split('/')[-1]
    currentpath = 'Data/IMG/' + filename
    image = ndimage.imread(currentpath)
    flipimage = np.fliplr(image)  # horizontally flip image
    origimages.append(image)
    origimages.append(flipimage)
    measurement = float(line[3])
    flipmeasurement = - measurement  # negative of original reading for flopped image
    origmeasurements.append(measurement)
    origmeasurements.append(flipmeasurement)

for line in lines:
    sourcepath = line[1]  # for left image
    filename = sourcepath.split('/')[-1]
    currentpath = 'data/IMG/' + filename
    image = ndimage.imread(currentpath)
    flipimage = np.fliplr(image)
    origimages.append(image)
    origimages.append(flipimage)
    measurement = float(line[3])
    measurement = measurement + correction  # correction added
    flipmeasurement = - measurement
    origmeasurements.append(measurement)
    origmeasurements.append(flipmeasurement)

for line in lines:
    sourcepath = line[2]  # for right image
    filename = sourcepath.split('/')[-1]
    currentpath = 'data/IMG/' + filename
    image = ndimage.imread(currentpath)
    flipimage = np.fliplr(image)
    origimages.append(image)
    origimages.append(flipimage)
    measurement = float(line[3])
    measurement = measurement - correction
    flipmeasurement = - measurement
    origmeasurements.append(measurement)
    origmeasurements.append(flipmeasurement)

origXTrain = np.array(origimages)
origYTrain = np.array(origmeasurements)

np.save('UXTrain1', origXTrain)
np.save('UYTrain1', origYTrain)
