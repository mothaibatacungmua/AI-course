from layer import \
    ConvPoolLayer, \
    HiddenLayer, \
    BatchNormalizationLayer, \
    DropoutLayer, \
    LogisticRegression

LAYER_TYPES = {
    'ConvPoolLayer': ConvPoolLayer,
    'HiddenLayer': HiddenLayer,
    'BatchNormalizationLayer': BatchNormalizationLayer,
    'DropoutLayer': DropoutLayer,
    'LogisticRegressionLayer': LogisticRegression
}