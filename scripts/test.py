class house(object):
	def __init__(self):
		self._testval = None

	@property
	def testval(self):
		return self._testval

	#@testval.setter
	#def testval(self, val):
	#	self._testval = val
