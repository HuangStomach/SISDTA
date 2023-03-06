class Hook:
    ecfps_sim = None

    def __init__(self):
        pass

    def record(self, module, input, output):
        self.ecfps_sim = output

    def output_dict(self):
        return self.ecfps_sim
