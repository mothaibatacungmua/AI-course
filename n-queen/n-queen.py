from random import randint
from backtracking import normal_backtracking, mrv_backtracking
from min_conflicts import min_conflicts
from hill_climbing import random_restart_hill_climbing
import time


class NQueen():
    def __init__(self, n, problem=[]):
        self.n = n
        self.problem = problem
        self.count = 0
        self.option = {}
        if len(problem) < n:
            self.problem = self.problem + [0]*(n - len(problem))

        if len(problem) > n:
            self.problem = self.problem[:n]

        pass

    def gen_rand_problem(self):
        p = [0] * self.n
        for i in range(0, self.n):
            p[i] = randint(0, self.n - 1)

        self.problem = p

        return p

    def check_collision(self, pos_a, pos_b):
        if pos_a == None or pos_b == None:
            return False

        if (pos_a[0] == pos_b[0]) or (pos_a[1] == pos_b[1]) or (abs(pos_a[0] - pos_b[0]) == abs(pos_a[1] - pos_b[1])):
            return True

        return False

    def check_collision_at(self, state, pos):
        for i in range(0, len(state)):
            if i != pos:
                if self.check_collision((i, state[i]), (pos, state[pos])):
                    return True
        return False

    def count_collision_at(self, state, pos):
        count = 0
        for i in range(0, len(state)):
            if i != pos:
                if self.check_collision((i, state[i]), (pos, state[pos])):
                    count = count + 1

        return count

    def count_collisions(self, state):
        count = 0
        for i in range(0, len(state)):
            for j in range(i + 1, len(state)):
                if self.check_collision((i, state[i]), (j, state[j])):
                    count = count + 1

        return count

    def is_goal_state(self, state):
        return self.count_collisions(state) == 0

    def init_domain_value(self):
        domain_value = [[0 for i in range(0, self.n)]for j in range(0, self.n)]

        for i in range(0, self.n):
            for j in range(0, self.n):
                domain_value[i][j] = (self.problem[i] + j) % self.n

        return domain_value

    def get_unassigned_vars(self, assignment):
        assigned_vars = list(map(lambda x: x[0], assignment))
        unassigned_vars = [x for x in range(0, self.n) if x not in assigned_vars]

        return assigned_vars, unassigned_vars

    def contraints(self, state, assigned_var, pos):
        for i in range(0, len(assigned_var)):
            if self.check_collision((assigned_var[i], state[assigned_var[i]]), (pos, state[pos])):
                return False
        return True

    def get_conflict_vars(self, state):
        conflicts = []
        for i in range(0, len(state)):
            for j in range(i+1, len(state)):
                if self.check_collision((i, state[i]), (j, state[j])):
                    if i not in conflicts and j not in conflicts:
                        conflicts.append(i)
                        conflicts.append(j)

        return conflicts

    def __normal_backtracking(self):
        return normal_backtracking(self)

    def __mrv_backtracking(self):
        if self.options == None:
            self.options = {'with_forward_checking':False, 'AC3':False}

        return mrv_backtracking(self)

    def __min_conflicts(self):
        if self.options == None:
            self.options = {'max_step':100000}

        return min_conflicts(self)

    def __random_restart_hill_climbing(self):
        if self.options == None:
            self.options = {'max_restart':1000}

        return random_restart_hill_climbing(self)

    NORMAL_BACKTRACKING = 0
    MRV_BACKTRACKING = 1
    MIN_CONFLICTS = 2
    RANDOM_RESTART_HILL_CLIMBING = 3
    def run(self, algorithms=NORMAL_BACKTRACKING, options=None):
        print '\n'
        print self.problem
        start_time = time.time()
        assignement = []
        goal = False
        self.options = options

        if algorithms == NQueen.NORMAL_BACKTRACKING:
            assignement, goal = self.__normal_backtracking()

        if algorithms == NQueen.MRV_BACKTRACKING:
            assignement, goal = self.__mrv_backtracking()

        if algorithms == NQueen.MIN_CONFLICTS:
            assignement, goal = self.__min_conflicts()

        if algorithms == NQueen.RANDOM_RESTART_HILL_CLIMBING:
            assignement, goal = self.__random_restart_hill_climbing()

        end_time = time.time()
        print assignement
        print goal, self.count
        print "--- %s seconds ---" % (end_time - start_time)
        pass

if __name__ == '__main__':
    queen_puzzle = NQueen(8)
    queen_puzzle.gen_rand_problem()
    queen_puzzle.run(algorithms=NQueen.NORMAL_BACKTRACKING)
    #queen_puzzle.run(algorithms=NQueen.MRV_BACKTRACKING)
    #queen_puzzle.run(algorithms=NQueen.MRV_BACKTRACKING, options={'with_forward_checking':True})
    #queen_puzzle.run(algorithms=NQueen.MRV_BACKTRACKING, options={'with_forward_checking': True, 'AC3':True})
    #queen_puzzle.run(algorithms=NQueen.MIN_CONFLICTS, options={'max_step':100000})
    queen_puzzle.run(algorithms=NQueen.RANDOM_RESTART_HILL_CLIMBING, options={'max_restart':10000})
    pass
