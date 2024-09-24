import math


class ReproducibleScheduler:
    def __init__(self, optimizer, strategy, last_epoch):
        """
        :param optimizer:
        :param strategy: LR decay strategy (string)
        :param last_epoch: the last epoch of training / the number of epochs
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.last_epoch = last_epoch

        self.strategy_function = self.get_strategy_function()
        self.base_lr = self.get_lr()

    def get_strategy_function(self):
        if self.strategy == 'cosine_annealing':
            return cosine_annealing
        elif self.strategy == 'constant':
            return constant

    def adjust_lr(self, current_epoch):
        current_lr = self.strategy_function(current_epoch, self.last_epoch) * self.base_lr

        for p in self.optimizer.param_groups:
            p['lr'] = current_lr

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def get_state(self):
        return {
            'strategy': self.strategy,
            'last_epoch': self.last_epoch,
            'base_lr': self.base_lr
        }

    def set_state(self, state):
        self.strategy = state['strategy']
        self.last_epoch = state['last_epoch']

        self.strategy_function = self.get_strategy_function()
        self.base_lr = state['base_lr']


def cosine_annealing(current_epoch, last_epoch):
    return 1e-2 + 0.5 * (1 - 1e-2) * (1 + math.cos(math.pi * current_epoch / last_epoch))

def constant(current_epoch, last_epoch):
    return 1
