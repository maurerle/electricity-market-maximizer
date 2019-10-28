import threading
from common.config import *
import common.config as conf
import time

class ProcessorTh(threading.Thread):
	def __init__(self, log):
		threading.Thread.__init__(self)
		self.name = 'processor'
		self.log = log
		self.running = False
		self.start()
		  
	def run(self):
		self.log.info("Processor Running")
		while True:
			if len(conf.HISTORY)!=0:
				self.running = True
				# File history managing
				fname = conf.HISTORY.pop(0)
				# Processing
				self.toDatabase(fname)
				# Clean folder
				os.remove(DOWNLOAD+'/'+fname)
			else:
				#print ("Here")
				if self.running:
					self.running = False
					self.log.info("Processing done")	
			time.sleep(.1)
			
	def toDatabase(self, fname):
		with open(DOWNLOAD+'/'+fname, 'r') as file:
			data = file.read()
		
		print ("[TODO] Processing {}".format(fname))

