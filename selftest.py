
import unittest
loader = unittest.TestLoader()
tests = loader.discover('./libs/.',pattern='*.py')
print ("test="+str(tests))
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
