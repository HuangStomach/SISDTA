import numpy as np

class Hook:
    record_dict = dict()

    def __init__(self, dataset='davis', sim_type='sis'):
        self.dataset = dataset
        self.sim_type = sim_type

    def record(self, key):
        def hook(module, input, output):
            self.record_dict[key] = output

        return hook

    def output_dict(self):
        res = dict()
        for key in self.record_dict.keys():
            res[key] = self.record_dict[key].detach().cpu().numpy()

        return res

    def save(self):
        d = self.output_dict()
        for key in self.record_dict.keys():
            np.savetxt('./output/{}/{}_{}.csv'.format(self.dataset, self.sim_type, key), 
                       d[key], fmt='%s', delimiter=',')
