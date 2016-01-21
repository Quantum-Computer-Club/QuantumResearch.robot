from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import linecache
import random

samples = linecache.getlines('svm.txt')
random.shuffle(samples)
alldata = ClassificationDataSet(16, 1, nb_classes=2)
for sample in samples:
    sample_array = sample.split('\t')[0:16]
    sample_result = sample.split('\t')[-1]
    for element in range(0, len(sample_array)):
        sample_array[element] = float(sample_array[element])
    sample_result = int(sample_result)
    alldata.addSample(sample_array, [sample_result])

tstdata, trndata = alldata.splitWithProportion( 0.25 )
print type(tstdata)
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

