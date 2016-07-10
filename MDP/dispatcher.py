class Dispatcher():
    def __init__(self):
        self.mapping_cb = {}
        pass

    def register(self, name, callback):
        self.mapping_cb[name] = callback
        pass

    def dispatch(self, name, *args, **kwargs):
        if self.mapping_cb.has_key(name):
            return self.mapping_cb[name](*args, **kwargs)

        return None
        pass