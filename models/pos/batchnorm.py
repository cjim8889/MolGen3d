from survae.transforms.bijections import BatchNormBijection1d



class BatchNormFlow(BatchNormBijection1d):
    def __init__(self, *args, **kwargs):
        super(BatchNormFlow, self).__init__(*args, **kwargs)

    def forward(self, x, mask=None, logs=None):
        return super(BatchNormFlow, self).forward(x)
    
    def inverse(self, z, mask=None):
        return super(BatchNormFlow, self).inverse(z), 0.