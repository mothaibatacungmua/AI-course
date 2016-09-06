import sys
import re
import random

# Credit: https://inst.eecs.berkeley.edu/~cs188/sp08/projects/blackjack/blackjack.py
# Note that this solution ignores naturals.

# A hand is represented as a pair (total, ace) where:
#  - total is the point total of cards in the hand (counting aces as 1)
#  - ace is true iff the hand contains an ace

# Return the empty hand.
def empty_hand():
    return (0, False)

# Return whether the hand has a useable ace
# Note that a hand may contain an ace, but it might not be useable
def hand_has_useable_ace(hand):
    total, ace = hand
    return ((ace) and (total + 10) <= 21)

# Return the value of the hand
# The value of the hand is either total or total + 10 (if the ace is useable)
def hand_value(hand):
    total, ace = hand
    if (hand_has_useable_ace(hand)):
        return (total + 10)
    else:
        return total


# Update a hand by adding a card to it
# Return the new hand
def hand_add_card(hand, card):
    total, ace = hand
    total += card
    if card == 1:
        ace = True;
    return (total,ace)

# Return the reward of the game (-1, 0, or 1) given the final player and dealer
# hands
def game_reward(player_hand, dealer_hand):
    player_val = hand_value(player_hand)
    dealer_val = hand_value(dealer_hand)

    if player_val > 21:
        return -1.0
    elif dealer_val > 21:
        return 1.0
    elif player_val < dealer_val:
        return -1.0
    elif player_val == dealer_val:
        return 0.0
    elif player_val > dealer_val:
        return 1.0


# Draw a card from an unbiased deck.
# Return the face value of the card (1 to 10)
def draw_card_unbiased():
    card = random.randint(1, 13)
    if card > 10:
        card = 10
    return card


# Draw a card from a biased deck rich in tens and aces
# Return the face value of the card (1 to 10)
def draw_card_biased():
    # choose ace with probability 11/130
    r = random.randint(1,130)
    if r <= 11:
        return 1
    # choose ten with probability 11/130
    if r <= 22:
        return 10
    # else choose 2-9, J, Q, K with equal probability
    card = random.randint(2, 12)
    if card > 10:
        card = 10
    return card


# Deal a player hand given the function to draw cards
# Return only the hand
def deal_player_hand(draw):
    hand = empty_hand()
    hand = hand_add_card(hand, draw())
    hand = hand_add_card(hand, draw())
    while hand_value(hand) < 11:
        hand = hand_add_card(hand, draw())

    return hand


# Deal the first card of a dealer hand given the function to draw cards.
# Return the hand and the shown card
def deal_dealer_hand(draw):
    hand = empty_hand()
    card = draw()
    hand = hand_add_card(hand, card)
    return hand, card


# Play the dealer hand using the fixed strategy for the dealer
# Return the resulting dealer hand
def play_dealer_hand(hand, draw):
    while hand_value(hand) < 17:
        hand = hand_add_card(hand, draw())
    return hand


# States are tuples (card, val, useable) where
# - card is the card the dealer is showing
# - val is the current value of the player's hand
# - useable is whether or not the player has a useable ace

# Action are either stay (False) or hit(True)
def select_random_state(all_states):
    n = len(all_states)
    r = random.randint(0, n-1)
    state = all_states[r]
    return state


# Select an action at random
def select_random_action():
    r = random.randint(1, 2)
    return (r == 1)


# Select the best action using current Q-values
def select_best_action(Q, state):
    if Q[(state, True)] > Q[(state, False)]:
        return True
    return False


# Select an action according to the epsilon-greddy policy
def select_e_greedy_action(Q, state, epsilon):
    r = random.random()
    if r < epsilon:
        return select_random_action()
    return select_best_action(Q, state)


# Given the state, return the player and dealer hand consistent with it
def hands_from_state(state):
    card, val, useable = state
    if useable:
        val -= 10
    player_hand = (val, useable)
    dealer_hand = empty_hand()
    dealer_hand = hand_add_card(dealer_hand, card)
    return card, dealer_hand, player_hand


# Given the dealer's card and player's hand, return state
def state_from_hands(card, player_hand):
    val = hand_value(player_hand)
    useable = hand_has_useable_ace(player_hand)
    return (card, val, useable)


# Return a list of the possible states
def state_list():
    states = []
    for card in range(1,11):
        for val in range(11, 21):
            states.append((card, val, False))
            states.append((card, val, True))
    return states


# Return a map of all (state, action) pairs -> values (initially zero)
def initialize_state_action_value_map():
    states = state_list()
    M = {}
    for state in states:
        M[(state, False)] = 0.0
        M[(state, True)] = 0.0
    return M


# Print a (state, action) -> value map
def print_state_action_value_map(M):
    for useable in [True, False]:
        if useable:
            print 'Useable ace'
        else:
            print 'No useable ace'
        print 'Values for staying'
        for val in range(21, 10, -1):
            for card in range(1, 11):
                print "%5.2f" % M[((card, val, useable), False)], ' ',
            print '| %d' % val
        print 'Values for hitting:'
        for val in range(21, 10, -1):
            for card in range(1, 11):
                print '%5.2f' % M[((card, val, useable), True)], ' ',
            print '| %d' % val
        print ' '


# Print the state-action-value function (Q)
def print_Q(Q):
    print '---- Q(s,a) ----'
    print_state_action_value_map(Q)


# Print the state-value function (V) given the Q-values
def print_V(Q):
    print '---- V(s) ----'
    for useable in [True, False]:
        if useable:
            print 'Usable ace'
        else:
            print 'No useable ace'
        for val in range(21, 10, -1):
            for card in range(1, 11):
                if Q[((card, val, useable), True)] > Q[((card, val, useable), False)]:
                    print '%5.2f' % Q[((card, val, useable), True)], ' ',
                else:
                    print '%5.2f' % Q[((card, val, useable), False)], ' ',
            print '| %d' % val
        print ' '


# Print a policy given the Q-values
def print_policy(Q):
    print '---- Policy ----'
    for useable in [True, False]:
        if useable:
            print 'Usable ace'
        else:
            print 'No useable ace'
        for val in range(21, 10, -1):
            for card in range(1, 11):
                if Q[((card, val, useable), True)] > Q[((card, val, useable), False)]:
                    print 'X',
                else:
                    print ' ',
            print '| %d' % val
        print ' '


# Initialize Q-values so that they produce the intial policy of sticking
# only 20 or 21
def initialize_Q():
    states = state_list()
    M = {}
    for state in states:
        card, val, useable = state
        if val < 20:
            M[(state, False)] = -0.001
            M[(state, True)] = 0.001
        else:
            M[(state, False)] = 0.001
            M[(state, True)] = 0.001

    return M


# Initialize number of times each (state, action) pair has been observed
def initialize_count():
    count = initialize_state_action_value_map()
    return count


# Monte-Carlo ES
#
# Run Monte-Carlo for the specified number of iterations and return Q-values
def monte_carlo_es(draw, n_iter):
    # initialize Q and observation count
    Q = initialize_Q()
    count = initialize_count()
    # get list of all states
    all_states = state_list()
    # iterate
    for n in range(0, n_iter):
        # initialize  list of (state, action) pairs encountered in episode
        sa_list = []
        # pick starting (state, action) using exploring starts
        state = select_random_state(all_states)
        action = select_random_action()
        dealer_card, dealer_hand, player_hand = hands_from_state(state)
        # update the (state, action) list
        sa_list.append((state, action))
        # generate the episode
        while action:
            player_hand = hand_add_card(player_hand, draw())
            if hand_value(player_hand) > 21:
                break
            state = state_from_hands(dealer_card, player_hand)
            action = select_best_action(Q, state)
            sa_list.append((state, action))
        # allow the dealer to play
        dealer_hand = play_dealer_hand(dealer_hand, draw)
        R = game_reward(player_hand, dealer_hand)
        # update Q using average return
        for sa in sa_list:
            Q[sa] = (Q[sa] * count[sa] + R)/ (count[sa] + 1)
            count[sa] += 1

    return Q


# Q-learning
#
# Run Q-learning for the specified number of iterations and return the Q-values
# In this implementation, alpha decreases over time
def q_learning(draw, n_iter, alpha, epsilon):
    # initialize Q and count
    Q = initialize_Q()
    count = initialize_count()
    # get list of all states
    all_states = state_list()
    # iterate
    for n in range(0, n_iter):
        # initialize state
        state = select_random_state(all_states)
        dealer_card, dealer_hand, player_hand = hands_from_state(state)
        # choose actions, update Q
        while(True):
            action = select_e_greedy_action(Q, state, epsilon)
            sa = (state, action)
            if action:
                # draw a card, update state
                player_hand = hand_add_card(player_hand, draw())
                # check if busted
                if hand_value(player_hand) > 21:
                    # update Q-value
                    count[sa] += 1.0
                    Q[sa] = Q[sa] + alpha/count[sa] * (-1.0 - Q[sa])
                    break
                else:
                    # update Q-value
                    s_next = state_from_hands(dealer_card, player_hand)
                    q_best = Q[(s_next, False)]
                    if Q[(s_next, True)] > Q[(s_next, False)]:
                        q_best = Q[(s_next, True)]
                    count[sa] += 1.0
                    Q[sa] = Q[sa] + alpha/count[sa] * (q_best - Q[sa])
                    # update state
                    state = s_next
            else:
                # allow the dealer to play
                dealer_hand = play_dealer_hand(dealer_hand, draw)
                # compute return
                R = game_reward(player_hand, dealer_hand)
                # update Q-value
                count[sa] += 1.0
                Q[sa] = Q[sa] + alpha/count[sa] * (R - Q[sa])

    return Q


# Sarsa
def sarsa(draw, n_iter, alpha, epsilon):
    # initialize Q and count
    Q = initialize_Q()
    count = initialize_count()
    # get list of all states
    all_states = state_list()
    # iterate
    for n in range(0, n_iter):
        # initialize state
        state = select_random_state(all_states)
        dealer_card, dealer_hand, player_hand = hands_from_state(state)
        # choose actions, update Q
        while (True):
            action = select_e_greedy_action(Q, state, epsilon)
            sa = (state, action)
            if action:
                # draw a card, update state
                player_hand = hand_add_card(player_hand, draw())
                # check if busted
                if hand_value(player_hand) > 21:
                    # update Q-value
                    count[sa] += 1.0
                    Q[sa] = Q[sa] + alpha / count[sa] * (-1.0 - Q[sa])
                    break
                else:
                    s_next = state_from_hands(dealer_card, player_hand)
                    action_ep = select_e_greedy_action(Q, s_next, epsilon)
                    q_eq = Q[(s_next, action_ep)]
                    count[sa] += + 1.0
                    Q[sa] = Q[sa] + alpha / count[sa] * (q_eq - Q[sa])
                    # update state
                    state = s_next
            else:
                # allow the dealer to play
                dealer_hand = play_dealer_hand(dealer_hand, draw)
                # compute return
                R = game_reward(player_hand, dealer_hand)
                # update Q-value
                count[sa] += 1.0
                Q[sa] = Q[sa] + alpha / count[sa] * (R - Q[sa])
    return Q


# Compute the expected value of the game using the learned Q-values or Sarsa
def expected_gain(draw, Q, n_iter):
    gain = 0.0
    for n in range(n_iter):
        player_hand = deal_player_hand(draw)
        dealer_hand = deal_dealer_hand(draw)
        state = state_from_hands(dealer_hand, player_hand)
        v = Q[(state, False)]
        if Q[(state, True)] > v:
            v = Q[(state, True)]
        gain = gain + v

    print 'Expected gain: %6.3f' % (gain / float(n_iter))
    print ' '

# main progam
# set parameters
if __name__ == '__main__':
   n_iter_mc = 10000000
   n_iter_q  = 10000000
   n_games = 100000
   alpha = 1
   epsilon = 0.1

   # run learning algorithms
   print 'MONTE CARLO ES -- UNBIASED DECK'
   Q = monte_carlo_es(draw_card_unbiased, n_iter_mc)
   print_Q(Q)
   print_V(Q)
   print_policy(Q)
   expected_gain(draw_card_unbiased, Q, n_games)
   print 'MONTE CARLO ES -- BIASED DECK'
   Q = monte_carlo_es(draw_card_biased, n_iter_q)
   print_Q(Q)
   print_V(Q)
   print_policy(Q)
   expected_gain(draw_card_biased, Q, n_games)

   print 'Q-LEARNING -- UNBIASED DECK'
   Q = q_learning(draw_card_unbiased, n_iter_q, alpha, epsilon)
   print_Q(Q)
   print_V(Q)
   print_policy(Q)
   expected_gain(draw_card_unbiased, Q, n_games)
   print 'Q-LEARNING -- BIASED DECK'
   Q = q_learning(draw_card_biased, n_iter_q, alpha, epsilon)
   print_Q(Q)
   print_V(Q)
   print_policy(Q)
   expected_gain(draw_card_biased, Q, n_games)

   print 'SASRA-LEARNING -- UNBIASED DECK'
   Q = sarsa(draw_card_unbiased, n_iter_q, alpha, epsilon)
   print_Q(Q)
   print_V(Q)
   print_policy(Q)
   expected_gain(draw_card_unbiased, Q, n_games)
   print 'SASRA-LEARNING -- BIASED DECK'
   Q = sarsa(draw_card_biased, n_iter_q, alpha, epsilon)
   print_Q(Q)
   print_V(Q)
   print_policy(Q)
   expected_gain(draw_card_biased, Q, n_games)