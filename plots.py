import matplotlib.pyplot as plt

def plotLC(accs, x, name):
	plt.plot(x, accs)
	plt.title(name)
	plt.ylabel('Accuracy')
	plt.xlabel('Iteration number')
	plt.savefig(name + '.png')
