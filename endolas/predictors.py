class _PredictorTemplate(object):
    def __init__(self):
        a = 1


class SegmentationPredictor(_PredictorTemplate):
    def __init__(self):
        super(SegmentationPredictor, self).__init__()

    def predict(self):
        print("Hello Segmentation")


class PeakfindingPredictor(_PredictorTemplate):
    def __init__(self):
        super(PeakfindingPredictor, self).__init__()

    def predict(self):
        print("Hello Peakfinding")


class RegistrationPredictor(_PredictorTemplate):
    def __init__(self):
        super(RegistrationPredictor, self).__init__()

    def predict(self):
        print("Hello Registration")


class NeighborPredictor(_PredictorTemplate):
    def __init__(self):
        super(NeighborPredictor, self).__init__()

    def predict(self):
        print("Hello Neighbor")


class PredictorContainer(object):
    def __init__(self):
        self._segmentation_predictor = SegmentationPredictor()
        self._peakfinding_predictor = PeakfindingPredictor()
        self._registration_predictor = RegistrationPredictor()
        self._neighbor_predictor = NeighborPredictor()

        self._predictors = [self._segmentation_predictor,
                            self._peakfinding_predictor,
                            self._registration_predictor,
                            self._neighbor_predictor]

    def predict_image(self, path):
        for predictor in self._predictors:
            predictor.predict()

    def predict_images(self, path):
        for predictor in self._predictors:
            predictor.predict()

    def predict_video(self, path):
        for predictor in self._predictors:
            predictor.predict()

