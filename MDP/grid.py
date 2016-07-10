from dispatcher import Dispatcher
from rectangle import MDPRectangle
from random import randint

class Grid(Dispatcher):
    @staticmethod
    def regcb(inst):
        inst.register("left-mouse-click", inst.cb_left_mouse_click)
        pass

    def __init__(self, canvas, x, y, num_x, num_y, rec_width, rec_height, reccls, rec_options={}):
        Dispatcher.__init__(self)
        self.canvas = canvas
        self.x = x
        self.y = y
        self.num_x = num_x
        self.num_y = num_y
        self.rec_width = rec_width
        self.rec_height = rec_height
        self.reccls = reccls
        self.rec_options = rec_options
        self.map_rec = {}
        Grid.regcb(self)

    def cb_left_mouse_click(self, *args, **kwargs):
        event = kwargs['event']
        pos_x = (event.x - self.x)/self.rec_width
        pos_y = (event.y - self.y)/self.rec_height
        if self.map_rec.has_key((pos_x, pos_y)):
            rec = self.map_rec[(pos_x, pos_y)]
            rec.dispatch('left-mouse-click', *args, **kwargs)
        pass

    def draw(self, *args, **kwargs):
        for i in range(0, self.num_x):
            for j in range(0, self.num_y):
                new_rec = self.reccls(self.canvas, i*self.rec_width + self.x, j*self.rec_height + self.y, self.rec_width, self.rec_height, self.rec_options, *args, **kwargs)
                new_rec.draw()
                self.map_rec[(i,j)] = new_rec

        pass


class MDPGrid(Grid):
    VALUE_ITER = 0
    POLICY_ITER = 1

    ACTION = {
        'LEFT':0,
        'RIGHT':1,
        'TOWARD':2
    }

    @staticmethod
    def regcb(inst):
        inst.register("left-mouse-click", inst.cb_left_mouse_click)
        inst.register("right-mouse-click", inst.cb_right_mouse_click)
        pass

    def __init__(self, canvas, x, y, num_x, num_y, rec_width, rec_height, rec_options={}, num_iter_id = None, text_iter = None):
        Grid.__init__(self,canvas, x, y, num_x, num_y, rec_width, rec_height, MDPRectangle, rec_options)
        MDPGrid.regcb(self)
        self.num_iter_id = num_iter_id
        self.start_point = None
        self.goal_point = None
        self.pit_point = None
        self.probs = None
        self.mode = MDPGrid.VALUE_ITER
        self.loop_count = 0
        self.stop = False
        self.walls = ()
        self.action_reward = 0.0
        self.discount_factor = 0.9
        self.policy = None
        self.text_iter = text_iter

    def reset(self):
        self.loop_count = 0
        self.stop = False
        self.walls = ()
        self.policy = None

    def config(self, probs={'LEFT':0.1, 'RIGHT':0.1, 'TOWARD':0.8}, mode = VALUE_ITER, action_reward = -0.04, discount_factor = 0.9):
        self.probs = probs
        self.mode = mode
        self.stop = False
        self.action_reward = action_reward
        self.discount_factor = discount_factor

        if mode == MDPGrid.VALUE_ITER:
            self.canvas.itemconfigure(self.text_iter, text="Value Iterations:")
        else:
            self.canvas.itemconfigure(self.text_iter, text="Policy Iterations:")
        pass

    def set_point(self, x, y, reward, color, sign=False):
        if not self.map_rec.has_key((x, y)):
            return False

        rec = self.map_rec[(x, y)]
        rec.change_value(reward, sign)
        rec.change_color(color)

        return True
        pass

    def set_start_point(self, x, y, color):
        if self.set_point(x, y, 0.0, color):
            self.start_point = (x, y)
        pass

    def set_goal_reward(self, x, y, reward, color):
        if self.set_point(x, y, reward, color, True):
            self.goal_point = (x, y)
        pass

    def set_pit_reward(self, x, y, reward, color):
        if self.set_point(x, y, reward, color):
            self.pit_point = (x, y)
        pass

    def set_wall(self, x, y, color):
        if self.set_point(x, y, None, color):
            self.walls = self.walls + ((x,y),)
        pass

    def get_successors(self, state):
        #left state
        left_state = (state[0]-1, state[1])

        #right state
        right_state = (state[0]+1, state[1])

        #toward state
        toward_state = (state[0], state[1]-1)

        #downward state
        downward_state = (state[0], state[1]+1)

        if not self.map_rec.has_key(left_state) or left_state in self.walls:
            left_state = state

        if not self.map_rec.has_key(right_state) or right_state in self.walls:
            right_state = state

        if not self.map_rec.has_key(toward_state) or toward_state in self.walls:
            toward_state = state

        if not self.map_rec.has_key(downward_state) or downward_state in self.walls:
            downward_state = state

        return {'WEST':left_state, 'EAST':right_state, 'SOUTH':downward_state, 'NORTH':toward_state}

    def calc_action_value(self, state, action, map_values):
        kw_successors = self.get_successors(state)

        ret = self.action_reward
        for k,v in kw_successors.iteritems():
            value = map_values[v]['value']
            if k == action:
                ret += self.discount_factor*self.probs['TOWARD']*value
            else:
                #There are a lot of if statements at here, I can make it more simple
                #but I want to remain them because it's eaiser to see with bellman equation
                if action == 'WEST' and k == 'EAST':
                    ret += self.discount_factor * 0.0 *value
                if action == 'WEST' and k == 'NORTH':
                    ret += self.discount_factor * self.probs['RIGHT'] * value
                if action == 'WEST' and k == 'SOUTH':
                    ret += self.discount_factor * self.probs['LEFT'] * value

                if action == 'EAST' and k == 'WEST':
                    ret += self.discount_factor * 0.0 * value
                if action == 'EAST' and k == 'NORTH':
                    ret += self.discount_factor * self.probs['LEFT'] * value
                if action == 'EAST' and k == 'SOUTH':
                    ret += self.discount_factor * self.probs['RIGHT'] * value

                if action == 'NORTH' and k == 'WEST':
                    ret += self.discount_factor * self.probs['LEFT'] * value
                if action == 'NORTH' and k == 'EAST':
                    ret += self.discount_factor * self.probs['RIGHT'] * value
                if action == 'NORTH' and k == 'SOUTH':
                    ret += self.discount_factor * 0.0 * value

                if action == 'SOUTH' and k == 'WEST':
                    ret += self.discount_factor * self.probs['RIGHT'] * value
                if action == 'SOUTH' and k == 'EAST':
                    ret += self.discount_factor * self.probs['LEFT'] * value
                if action == 'SOUTH' and k == 'NORTH':
                    ret += self.discount_factor * 0.0 * value

        return ret
        pass

    def calc_state_value(self, state, map_values):
        map_rec = self.map_rec[state]
        if state == self.goal_point or state == self.pit_point:
            return map_rec.value
        l = []
        for action in ['NORTH', 'EAST', 'SOUTH', 'WEST']:
            l.append((self.calc_action_value(state, action, map_values), action))

        l = sorted(l, key=lambda e: e[0], reverse=True)

        #if(state == (2,1)):
        #    print l

        return l[0]
        pass

    def copy_map_values(self):
        copy_map = {}
        for i in range(0, self.num_x):
            for j in range(0, self.num_y):
                copy_map[(i,j)] = {'value':self.map_rec[(i,j)].value, 'dir':self.map_rec[(i,j)].direction}

        return copy_map

    def run_value_iteration(self):
        copy_map = self.copy_map_values()
        for i in range(0, self.num_x):
            for j in range(0, self.num_y):
                if (i,j) == self.goal_point or (i,j) == self.pit_point or (i,j) in self.walls:
                    continue

                map_rec = self.map_rec[(i,j)]
                #update value and policy
                value, action = self.calc_state_value((i,j),copy_map)
                map_rec.change_value(round(value,4))
                map_rec.change_direction(action)

        pass

    def copy_policy(self, policy):
        clone = {}
        for i in range(0, self.num_x):
            for j in range(0, self.num_y):
                clone[(i,j)] = {'value':policy[(i,j)]['value'], 'dir':policy[(i, j)]['dir']}

        return clone

    def policy_evaluation(self, policy, loop):
        for i in range(0, loop):
            clone = self.copy_policy(policy) # so stupid
            for j in range(0, self.num_x):
                for k in range(0, self.num_y):
                    if (j, k) == self.goal_point or (j, k) == self.pit_point or (j, k) in self.walls:
                        continue
                    policy[(j,k)]['value'] = self.calc_action_value((j,k), policy[(j,k)]['dir'], clone)


        return policy
        pass

    def random_init_policy(self):
        dirs = ['NORTH', 'EAST', 'SOUTH', 'WEST']
        policy = {}
        for i in range(0, self.num_x):
            for j in range(0, self.num_y):
                policy[(i, j)] = {'value':self.map_rec[(i, j)].value, 'dir':dirs[randint(0,3)]}

        return policy

    def run_policy_iteration(self):
        self.policy_evaluation(self.policy, 30)
        clone = self.copy_policy(self.policy)

        for i in range(0, self.num_x):
            for j in range(0, self.num_y):
                if (i, j) == self.goal_point or (i, j) == self.pit_point or (i, j) in self.walls:
                    continue

                map_rec = self.map_rec[(i, j)]
                map_rec.change_value(round(self.policy[(i,j)]['value'], 4))
                map_rec.change_direction(self.policy[(i,j)]['dir'])

                value, action = self.calc_state_value((i, j), clone)
                if value > self.policy[(i,j)]['value']:
                    self.policy[i,j]['dir'] = action

        pass

    def cb_left_mouse_click(self, *args, **kwargs):
        if self.mode == MDPGrid.VALUE_ITER:
            self.run_value_iteration()

        if self.mode == MDPGrid.POLICY_ITER:
            if self.policy == None:
                self.policy = self.random_init_policy()
            self.run_policy_iteration()

        self.loop_count += 1
        self.canvas.itemconfigure(self.num_iter_id, text = str(self.loop_count))

    def cb_right_mouse_click(self, *args, **kwargs):
        self.reset()
        if kwargs.has_key('event'):
            kwargs.pop('event', None)

        self.draw(*args, **kwargs)
        self.config()
        self.set_start_point(0, 2, 'blue')
        self.set_goal_reward(3, 0, 1, 'green')
        self.set_pit_reward(3, 1, -1, 'red')
        self.set_wall(1, 1, 'gray')