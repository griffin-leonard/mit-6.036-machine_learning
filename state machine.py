from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        o = []
        s = self.start_state
        for x in input_seq:
            s = self.transition_fn(s, x)
            out = self.output_fn(s)
            o.append(out)
        return o


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0,False)

    def transition_fn(self, s, x):
        n = sum(x)
        if s[1]:
            if n == 0: return (1,False)
            elif n == 1: return (0,True)
            elif n == 2: return (1,True)
        else:
            if n == 0: return (0,False)
            elif n == 1: return (1,False)
            elif n == 2: return (0,True)


    def output_fn(self, s):
        return s[0]


class Reverser(SM):
    start_state = None
    end = False
    seq1 = []
    i = 0

    def transition_fn(self, s, x):
        if self.end or x == 'end':
            self.end = True
            self.i -= 1
            try: return self.seq1[self.i]
            except: return None
        elif not self.end:
            self.seq1.append(x)
            return None

    def output_fn(self, s):
        return s


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state):
        self.Wsx, self.Wss = Wsx, Wss
        self.Wo, self. Wss0, self.Wo0 = Wo, Wss_0, Wo_0
        self.f1, self.f2 = f1, f2
        self.start_state = start_state

    def transition_fn(self, s, x):
        return self.f1(self.Wss@s + self.Wsx@x + self.Wss0)

    def output_fn(self, s):
        return self.f2(self.Wo@s + self.Wo0)
