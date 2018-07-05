class DataReader:

	def __init__(self):
		self.img_fd = 0
		self.label_fd = 0
		self.currentImage = 0
		self.images = []
		self.labels = []

		self.initializeReading()
		self.dataToBuf()

	def initializeReading(self):
		self.img_fd = open('../numbers/train-images.idx3-ubyte','r')
		self.img_fd.read(16)
		self.label_fd = open('../numbers/train-labels.idx1-ubyte','r')
		self.label_fd.read(8)

	def dataToBuf(self):
		number_buf = self.img_fd.read(784 * 60000)
		labels_buf = self.label_fd.read(60000)
		for i in range(0, 60000):
			self.images.append(number_buf[(i * 784):((i + 1) * 784)])
			self.labels.append(ord(labels_buf[i]))

	def getImage(self, image):
		imageArrayFloat = []
		for i in range(0, 784):
			imageArrayFloat.append(float(ord(self.images[image][i])) / 255.0)
		return(imageArrayFloat)

	def getLabel(self, label):
		return(self.labels[label])

	def printLabel(self, label):
		print ("Expected result: " + str(self.labels[label]))
