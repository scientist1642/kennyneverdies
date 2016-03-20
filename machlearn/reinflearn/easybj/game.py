# Reinforcement learning Task 1
# Zura Isakadze

import copy
import random
import numpy as np


class State(object):
    def __init__(self):
        self.pl_score = 0
        self.dl_score = 0
        self.terminal = False


class Actor:
    # enum
    dealer = "dealer"
    player = "player"


class Action:
    # enum
    stick = 0
    hit = 1


class Game(object):
    def __init__(self):
        pass

    def draw_card(self):
        num = random.randint(1, 10)
        if random.randint(1, 3) == 1:
            num *= -1
        return num

    def debug_card(self, num, actor):
        if not self.step_debug:
            return
        col = 'b' if num > 0 else 'r'
        num = abs(num)
        print '{} received card ({}, {})'.format(actor, col, num)

    def debug_score(self, actor, score):
        if not self.step_debug:
            return
        print "{}'s new score is {}".format(actor, score)
    
    def debug_result(self, res):
        if not self.step_debug:
            return
        # -1  player lost, 0 draw,  1 player won
        if res == 0:
            print 'Game was Drawn!'
        elif res == -1:
            print 'Player Lost'
        else:
            print 'Player won'
    
    def debug_msg(self, msg):
        if not self.step_debug:
            return
        print msg


    def step(self, state, action, debug=False):
        """ draw card and reutrn a tuple of new state and reward"""
        
        self.step_debug = debug

        if state.terminal:
            raise ('Action initiated while in the terminal state')
        

        nstate = copy.deepcopy(state) # nstate is a new state
        reward = 0
        if action == Action.hit:
            self.debug_msg('Player hits')
            card = self.draw_card()
            self.debug_card(card, Actor.player)
            nstate.pl_score += card
            self.debug_score(Actor.player, nstate.pl_score)
            if not 1 <= nstate.pl_score <= 21:
                # player goes bust
                self.debug_msg('Player goes bust')
                reward = -1
                self.debug_result(-1)
                nstate.terminal = True

        elif action == Action.stick:
            self.debug_msg('Player sticks')
            # it's now dealer's turn

            nstate.terminal = True
            # we end up in terminal stage one way or enouther
            while 1 <= nstate.dl_score < 17:
                # dealer decides to hit
                card = self.draw_card()
                self.debug_card(card, Actor.dealer)
                nstate.dl_score += card
                self.debug_score(Actor.dealer, nstate.dl_score)

            if not 1 <= nstate.dl_score <= 21:
                self.debug_msg('Dealer goes bust')
                reward = 1
            else:
                self.debug_msg('Dealer sticks')
                if nstate.dl_score > nstate.pl_score:
                    reward = -1
                elif nstate.dl_score == nstate.pl_score:
                    reward = 0
                else:
                    reward = 1
                # reward right now denotes who won as well
            
            self.debug_result(reward)

        return (nstate, reward)


    def monte_carlo_control(self, nepisodes, n0):
        ''' e-greedy monte carlo control  '''
        
        # init value function to zero, note we don't need to
        # explicitely have pi as value function defined greedy policy
        # ep for each timestamp = n0 / (nst + cnt(st))

        q = np.zeros((22, 22, 2))  # 0..21 pl value, 0..21 deal value, 0..1 actions
        cnt = np.zeros((22, 22, 2)) # count for each pair
        for k in xrange(nepisodes):
            # sample episode using updated policy
            episode = [] # (s1, A1, R2) (s2, A2, R3) ...  
            state = State()
            state.pl_score = abs(self.draw_card())
            state.dl_score = abs(self.draw_card())
            while not state.terminal:
                # with ep probability we choose random action
                #import ipdb; ipdb.set_trace()
                num_of_visits = sum(cnt[state.pl_score, state.dl_score, :])
                ep = float(n0) / (n0 + num_of_visits)

                if np.random.binomial(1, ep):
                    action = np.random.choice(q.shape[2]) # random action
                else:
                    action = np.argmax(q[state.pl_score, state.dl_score])
                
                cnt[state.pl_score, state.dl_score, action] += 1
                

                nstate, rew = self.step(state, action)
                episode.append((state, action, rew))
                state = nstate

            # G (return) for each (s, a) is a final reward
            # update value function / policy
            frew = episode[-1][2]
            for (st, act, _) in episode:
                x = st.pl_score
                y = st.dl_score
                alpha = 1. / cnt[x, y, act]
                q[x, y, act] += alpha * (frew - q[x, y, act])

        return q
    
    def e_greedy_act(self, s, cnt, n0, q):
        num_of_visits = sum(cnt[s.pl_score, s.dl_score, :])
        ep = float(n0) / (n0 + num_of_visits)
        if np.random.binomial(1, ep):
            return np.random.choice(q.shape[2]) # random action
        else:
            return np.argmax(q[s.pl_score, s.dl_score])
    
    def mse_error(self, a, b):
        # mean squared error of two matricies
        return ((a - b) ** 2).mean()

    def sarsa_lambda_control(self, nepisodes, n0, lmbd, q_star=None, mse_prog=False):
        """ 
        returns pair of (q, mse_array)
        runs sarsa for nepisodes, lmbd - lambda, q_star: policy to calculate
        mean square error for, if mse_prog is true returns all mse for each 
        episode, otherwise returns only the last one 
        """
        
        q = np.ones((22, 22, 2))  # 0..21 pl value, 0..21 deal value, 0..1 actions
        q.fill(0.5) # just to increase initial mse error
        e = np.zeros((22, 22, 2), dtype=float) # eligibility traces
        cnt = np.ones((22, 22, 2)) # count for each pair, ones to avoid 0 div
        mses = [] #mean squeared errors
        for k in xrange(nepisodes):
            # sample episode using updated policy
            #import ipdb; ipdb.set_trace()
            s = State()
            s.pl_score = abs(self.draw_card())
            s.dl_score = abs(self.draw_card())
            if mse_prog: # mse progress for each episode
                mses.append(self.mse_error(q_star, q))
            e.fill(0)
            # choose the initial action with e-greedy policy as well
            a = self.e_greedy_act(s, cnt, n0, q)
            while not s.terminal:
                s_prime, r = self.step(s, a)

                if not s_prime.terminal:
                    a_prime = self.e_greedy_act(s_prime, cnt, n0, q)
                    nextq = q[s_prime.pl_score, s_prime.dl_score, a_prime]
                else:
                    # a_prime, nextq not important
                    a_prime, nextq = 0, 0

                delta = r + nextq - q[s.pl_score, s.dl_score, a]

                cnt[s.pl_score, s.dl_score, a] += 1
                e[s.pl_score, s.dl_score, a] += 1
                q +=  1. / cnt * delta * e
                e *= lmbd

                s, a = s_prime, a_prime

        if not q_star is None:
            # we return a pair of action policy and mse_error
            mses.append(self.mse_error(q_star, q))
            return (q, mses)

        return q

    
    def sarsa_lambda_control_lfa(self, nepisodes, st_size, exp_rate, lmbd, q_star=None, mse_prog=False):
        """ 
        returns pair of (q, mse_array)
        runs sarsa for nepisodes, lmbd - lambda, q_star: policy to calculate
        mean square error for, if mse_prog is true returns all mse for each 
        episode, otherwise returns only the last one 
        uses linear function approximation
        er - exploration rate
        """
        
        def stateact_to_feature(pl, dl, act):
            pls = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
            dls = [(1, 4), (4, 7), (7, 10)]
            ft = np.zeros(11)
            for i in range(6):
                if pls[i][0] <= pl <= pls[i][1]:
                    ft[i] = 1
            for i in range(3):
                if dls[i][0] <= dl <= dls[i][1]:
                    ft[i + 6] = 1
            if act == 0:
                ft[9] = 1
            else:
                ft[10] = 1
            return ft
        
        def get_val(w, pl, dl, act):
            features = stateact_to_feature(pl, dl, act)
            return np.inner(features, w)

        def calc_mse(w):
            # TODO remove for loops later
            ret = 0
            for ind, v in np.ndenumerate(q_star):
                ret += (get_val(w, *ind) -  v) ** 2
            return ret / q_star.size

        def e_greedy(s, w):
            if np.random.binomial(1, exp_rate):
                return np.random.choice(2) # random action
            else:
                return np.argmax([get_val(w, s.pl_score, s.dl_score, 0), 
                        get_val(w, s.pl_score, s.dl_score, 1)])

        q = np.ones((22, 22, 2))  # 0..21 pl value, 0..21 deal value, 0..1 actions
        #q.fill(0.5) # just to increase initial mse error
        w = np.zeros(11) # feature weights
        w.fill(0.5) 
        # just to increase initial mse error, correct weights are 
        # close to 0

        e = np.zeros((22, 22, 2), dtype=float) # eligibility traces
        #cnt = np.ones((22, 22, 2)) # count for each pair, ones to avoid 0 div
        mses = [] #mean squeared errors
        for k in xrange(nepisodes):
            # sample episode using updated policy
            #import ipdb; ipdb.set_trace()
            s = State()
            s.pl_score = abs(self.draw_card())
            s.dl_score = abs(self.draw_card())
            if mse_prog: # mse progress for each episode
                #import ipdb; ipdb.set_trace() 
                mses.append(calc_mse(w))
            #e.fill(0)
            et = 0
            # choose the initial action with e-greedy policy as well
            #a = self.e_greedy_act(s, cnt, n0, q)
            a = e_greedy(s, w)
            while not s.terminal:
                s_prime, r = self.step(s, a)

                if not s_prime.terminal:
                    #a_prime = self.e_greedy_act(s_prime, cnt, n0, q)
                    a_prime = e_greedy(s_prime, w)
                    #nextq = q[s_prime.pl_score, s_prime.dl_score, a_prime]
                    nextq = get_val(w, s_prime.pl_score, s_prime.dl_score, a_prime)
                else:
                    # a_prime, nextq not important
                    a_prime, nextq = 0, 0

                #delta = r + lmbd * nextq - q[s.pl_score, s.dl_score, a]
                curq = get_val(w, s.pl_score, s.dl_score, a)

                delta = r + nextq - curq
                et = lmbd * et + stateact_to_feature(s.pl_score, s.dl_score, a)
                dew = st_size * delta * et
                w += dew   # gradient descend 
                #cnt[s.pl_score, s.dl_score, a] += 1
                #e[s.pl_score, s.dl_score, a] += 1
                #q +=  1 / cnt * delta * e
                #e *= lmbd

                s, a = s_prime, a_prime

        
        # we can build up q from w just for testing purposes
        # normally we would only need w

        for ind, v in np.ndenumerate(q):
            q[ind] = get_val(w, *ind)

        if not q_star is None:
            # we return a pair of q and mse_error
            mses.append(calc_mse(w))
            return (q, mses)
            
        return q

    def evaluate_policy_naive(self, pi, nepisodes):
        ''' evaluate the given policy pi S x S, and return expected reward
            of the game. It just simulates the game for n time and returns 
            the mean.
        '''
        
       
        reward_sum = 0
        for episode in xrange(nepisodes):
            state = State()
            state.pl_score = abs(self.draw_card())
            state.dl_score = abs(self.draw_card())
            while not state.terminal:
                action = pi[state.pl_score, state.dl_score]
                state, rew = self.step(state, action, debug=False)
            reward_sum += rew

        return reward_sum / float(nepisodes)


    def run(self):
        state = State()
        state.pl_score = abs(self.draw_card())
        state.dl_score = abs(self.draw_card())
        print 'player starts with {}'.format(state.pl_score)
        print 'dealer starts with {}'.format(state.dl_score)
        while not state.terminal:
            if random.randint(1, 3) <=2: 
                action = Action.hit
            else:
                action = Action.stick
            state, rew = self.step(state, action, debug=False)
            print 'reward is {}'.format(rew)


game = Game()
#game.monte_carlo_control(10000, 100)

#q = game.sarsa_lambda_control(nepisodes=1000000, n0=100, lmbd=1, q_star=None, mse_prog=True)
#q = game.sarsa_lambda_control_lfa(nepisodes=10, st_size=0.01, exp_rate=0.05, lmbd=1)
