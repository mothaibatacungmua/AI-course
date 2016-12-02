class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        if not conf.stack or not conf.buffer:
            return -1

        if conf.stack[0] == 0 and len(conf.stack)==1:
            return -1
        for arc in conf.arcs:
            if conf.stack[-1] == arc[2]:
                return -1

        idx_wi = conf.stack.pop()
        idx_wj = conf.buffer[0]

        conf.arcs.append((idx_wj, relation, idx_wi))

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        #raise NotImplementedError('Please implement reduce!')
        #return -1

        if not conf.stack:
            return -1

        for arc in conf.arcs:
            if conf.stack[-1] == arc[2]:
                conf.stack.pop()
                return

        return -1


    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        #raise NotImplementedError('Please implement shift!')
        #return -1

        if not conf.buffer or len(conf.buffer) == 0:
            return -1

        idx_wj = conf.buffer.pop(0)
        conf.stack.append(idx_wj)
