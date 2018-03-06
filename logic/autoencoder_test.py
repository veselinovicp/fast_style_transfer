import unittest

import logic.autoencoder as lg


class TestMethods(unittest.TestCase):


    # def test_autoencoder(self):
    #     autoencoder = lg.AutoEncoder(train_size=10, batch_size=5, input_images = '../data/michelangelo', dimension=500)
    #
    #     autoencoder.train()

    def test_vgg(self):
        autoencoder = lg.AutoEncoder(train_size=2, batch_size=5, input_images = '../data/michelangelo_2', dimension=500)
        # autoencoder.predict_test_value("../output/weights.hdf5", image_path="../data/michelangelo/class 0/Battle_of_CascinaStudy_of_a_Man1504.jpg", detect_edges=True)
        autoencoder.predict_test_value("../output/weights.hdf5", image_path="../data/input_image.png", detect_edges=True)




if __name__ == '__main__':
    unittest.main()
