class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the accuracy score"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0
        self.correct_count = 0
        self.total_count = 0

    def update(self, n_correct, n):
        self.correct_count += n_correct
        self.total_count += n
        self.acc = self.correct_count / self.total_count
