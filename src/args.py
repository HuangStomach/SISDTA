import argparse

class Args:
    def __init__(self, action='train'):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--device', default='cpu', type=str, metavar=None, 
            help='Name of the processor used for computing')
        self.parser.add_argument('-d', '--dataset', default='kiba', type=str, metavar='[kiba, davis]', 
            help='Name of the selected data set')
        self.parser.add_argument('--sim-type', default='sis', type=str, metavar=None,
            help='Similarity Strategy')
        self.parser.add_argument('-dt', '--d_threshold', default=0.7, type=float, metavar=None,
            help='Thresholds for drug relationship graphs')
        self.parser.add_argument('-pt', '--p_threshold', default=0.7, type=float, metavar=None,
            help='Thresholds for protein relationship graphs')

        if action == 'train':
            self.parser.add_argument('-s', '--setting', default=1, type=int, metavar=None, 
                help='Experimental setting')
            self.parser.add_argument('-e', '--epochs', default=1000, type=int, metavar=None, 
                help='Number of training iterations required')
            self.parser.add_argument('-b', '--batch-size', default=512, type=int, metavar=None,
                help='Size of each training batch')
            self.parser.add_argument('-lr', '--learning-rate', default=0.002, type=float, metavar=None,
                help='The step size at each iteration')
            self.parser.add_argument('-l1', '--lambda_1', default=1, type=float, metavar=None,
                help='AutoEncoder loss function weights')
            # parser.add_argument('-l2', '--lambda_2', default=0.00001, type=float, metavar='float')
            # parser.add_argument('-u', '--unit', default=0.1, type=float, metavar='float', help='unit of target')
    
    def parser(self):
        return self.parser

    def parse_args(self):
        return self.parser.parse_args()
    
    def print(self):
        print(self.parse_args())
        return self
