class DefenseBase:
    def __init__(self, args):
        self.args = args

    def train(self):
        raise NotImplementedError

    def verify(self):
        raise NotImplementedError

    def run(self):
        self.train()
        self.verify()

