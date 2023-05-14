
import unittest
import pytest
import glob
import os
listOfFile=glob.glob("./libs/*.py")
paramaters = ["--junitxml","results.xml","--capture=no"]
for i in listOfFile:
    paramaters.append(i)
print ("run tests with parameters ="+str(paramaters))
pytest.main(paramaters)


"""
loader = unittest.TestLoader()
tests = loader.discover('./libs/.',pattern='*.py')
print ("test="+str(tests))
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
"""