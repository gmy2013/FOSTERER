import torch
from bisect import bisect_right
import math



class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=0):
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(optimizer, torch.optim.Optimizer) :
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr


class _WarmUpLRScheduler(_LRScheduler):


    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=0):
        assert warmup_steps >= 2 or warmup_steps == 0
        if warmup_steps == 0:
            assert base_lr == warmup_lr
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps >= 2 and self.last_iter < self.warmup_steps:
            target_lr = (self.warmup_lr - self.base_lr) / \
                (self.warmup_steps-1) * (self.last_iter-1) + self.base_lr
            scale = target_lr / self.base_lr
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None


class StepLRScheduler(_WarmUpLRScheduler):


    def __init__(self, optimizer, lr_steps, lr_mults, base_lr, warmup_lr, warmup_steps, max_iter, last_iter=0):
        super(StepLRScheduler, self).__init__(
            optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(lr_steps) == len(
            lr_mults), "{} vs {}".format(lr_steps, lr_mults)
        for x in lr_steps:
            assert isinstance(x, int)
        if not list(lr_steps) == sorted(lr_steps):
            raise ValueError('lr_steps should be a list of'
                             ' increasing integers. Got {}', lr_steps)
        self.lr_steps = lr_steps
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1]*x)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.lr_steps, self.last_iter)
        scale = self.warmup_lr*self.lr_mults[pos] / self.base_lr
        return [base_lr*scale for base_lr in self.base_lrs]


Step = StepLRScheduler


class StepDecayLRScheduler(_WarmUpLRScheduler):


    def __init__(self, optimizer, step_size, decay, base_lr, warmup_lr, warmup_steps, max_iter, last_iter=0):
        super(StepDecayLRScheduler, self).__init__(
            optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        self.step_size = step_size
        self.decay = decay

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        num = (self.last_iter - self.warmup_steps) // self.step_size
        scale = self.decay ** num * self.warmup_lr / self.base_lr
        return [base_lr*scale for base_lr in self.base_lrs]


StepDecay = StepDecayLRScheduler


class CosineLRScheduler(_WarmUpLRScheduler):


    def __init__(self, optimizer, max_iter, min_lr, base_lr, warmup_lr, warmup_steps, last_iter=0):
        super(CosineLRScheduler, self).__init__(
            optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.max_iter = max_iter
        self.min_lr = min_lr

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        step_ratio = (self.last_iter-self.warmup_steps) / \
            (self.max_iter-self.warmup_steps)
        target_lr = self.min_lr + \
            (self.warmup_lr - self.min_lr) * \
            (1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.base_lr
        return [scale*base_lr for base_lr in self.base_lrs]


Cosine = CosineLRScheduler


class PolynomialLRScheduler(_WarmUpLRScheduler):


    def __init__(self, optimizer, power, max_iter, base_lr, warmup_lr, warmup_steps, last_iter=0):
        super(PolynomialLRScheduler, self).__init__(
            optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.max_iter = max_iter
        self.power = power

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        factor = (1 - (self.last_iter-self.warmup_steps) /
                  float(self.max_iter)) ** self.power
        target_lr = factor * self.warmup_lr
        scale = target_lr / self.base_lr
        return [scale*base_lr for base_lr in self.base_lrs]


Poly = PolynomialLRScheduler
