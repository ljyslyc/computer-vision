import unittest
import operations
import utils

class TestProj3(unittest.TestCase):

    def test_gaussMake(self):
        x = operations.makeGaussKernel(lowSig = 1)
        print(x)

    # def test_grayscale2RGB(self):
    #     utils.grayscale2RGB(im)

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

    
if __name__ == '__main__':
    unittest.main()
    Test_gas