# import logic.StyleTransfer as lg

import logic.autoencoder as lg


def train():

    # style_transfer_1 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg", output_path ="../output/1.png",
    #                                   style_image_path="../data/picasso.jpg", iterations=10, style_weight=10)
    #
    # style_transfer_1.transfer()
    #
    # style_transfer_2 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg", output_path ="../output/2.png",
    #                                   style_image_path="../data/wave.jpg", iterations=10, style_weight=10)
    #
    # style_transfer_2.transfer()
    #
    # style_transfer_3 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg", output_path ="../output/3.png",
    #                                   style_image_path="../data/van_gough.jpg", iterations=10, style_weight=10)
    #
    # style_transfer_3.transfer()
    #
    # style_transfer_4 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg", output_path ="../output/4.png",
    #                                     style_image_path="../data/block.jpg", iterations=10, style_weight=10)
    #
    # style_transfer_4.transfer()
    #
    # style_transfer_5 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg", output_path ="../output/5.png",
    #                                     style_image_path="../data/marilyn.jpg", iterations=10, style_weight=10)
    #
    # style_transfer_5.transfer()

        autoencoder = lg.AutoEncoder(train_size=1000, batch_size=32, input_images = '../data/michelangelo', dimension=500)

        autoencoder.train()


if __name__ == '__main__':
    train()
