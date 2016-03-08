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
game.monte_carlo_control(10000, 100)

