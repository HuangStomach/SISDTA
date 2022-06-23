from kiba.kiba import Kiba

handlers = {
    'kiba': Kiba
}
class Data:
    def __init__(self, type = 'kiba'):
        self.handler = handlers[type]

    