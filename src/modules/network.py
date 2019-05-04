import chainer
import chainer.links as L
import chainer.functions as F
    
class GeneratingStage(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.vgg = L.VGG16Layers()
            self.dc1 = L.Deconvolution2D(512, 256, ksize=2, stride=2)
            self.dc2 = L.Deconvolution2D(256, 64, ksize=2, stride=2)
            self.dc3 = L.Deconvolution2D(64, 16, ksize=2, stride=2)

    def forward(self, x):
        h = self.vgg(x, layers=['pool5'])
        h = F.relu(self.dc1(h['pool5']))
        h = F.relu(self.dc2(h))
        out = F.relu(self.dc3(h))

        return out

class RefinementStage(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            # Encoder for image data.
            self.vgg = L.VGG16Layers()
            # Encoder for label data.
            self.co1 = L.Convolution2D(16, 64, ksize=2, stride=2)
            self.co2 = L.Convolution2D(64, 256, ksize=2, stride=2)
            self.co3 = L.Convolution2D(256, 512, ksize=2, stride=2)
            # decoder
            self.dc1 = L.Deconvolution2D(1024, 256, ksize=2, stride=2)
            self.dc2 = L.Deconvolution2D(256, 64, ksize=2, stride=2)
            self.dc3 = L.Deconvolution2D(64, 16, ksize=2, stride=2)

    def forward(self, image, label):
        h_image = self.vgg(image, layers=['pool5'])

        h_label = F.relu(self.co1(label))
        h_label = F.relu(self.co2(h_label))
        h_label = F.relu(self.co3(h_label))

        h = F.concat((h_image['pool5'], h_label))
        h = F.relu(self.dc1(h))
        h = F.relu(self.dc2(h))
        out = F.relu(self.dc3(h))

        return out
