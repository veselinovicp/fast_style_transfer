import unittest

import logic.DataMachine as lg


class TestMethods(unittest.TestCase):

    def test_simple(self):
        data_machine = lg.DataMachine(input_image="../data/Satyrs_HeadBrown_InkLouvre.jpg")
        data_machine.prepare_data()


if __name__ == '__main__':
    unittest.main()
