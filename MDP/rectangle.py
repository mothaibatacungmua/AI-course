from dispatcher import Dispatcher
import math
import numpy as np

class Rectangle(Dispatcher):
    @staticmethod
    def regcb(inst):
        pass

    def __init__(self, canvas, x, y, width, height, options={}):
        Dispatcher.__init__(self)
        self.obj_id = None
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.options = options

    def draw(self):
        self.obj_id = self.canvas.create_rectangle(self.x, self.y, self.x + self.width, self.y + self.height, self.options)
        return self.obj_id

    def get_obj_id(self):
        return self.obj_id


class TextRectangle(Rectangle):
    @staticmethod
    def regcb(inst):
        inst.register("change-data", inst.cb_change_data)
        pass

    def __init__(self, canvas, x, y, width, height, options={}, text_options={}):
        Rectangle.__init__(self, canvas, x, y, width, height, options)
        self.text_id = None
        self.text_options = text_options
        TextRectangle.regcb(self)

    def draw(self):
        Rectangle.draw(self)
        self.text_id = self.canvas.create_text(self.x+self.width/2, self.y+self.height/2, self.text_options)

    def change_text(self, text):
        self.text_options['text'] = text
        self.canvas.itemconfigure(self.text_id, text=text)

    def cb_change_data(self, *args, **kwargs):
        pass


class MouseRectangle(TextRectangle):
    @staticmethod
    def regcb(inst):
        inst.register("left-mouse-click", inst.cb_left_mouse_click)
        pass

    def __init__(self, canvas, x, y, width, height, options={}, text_options={}):
        self.count = 0
        text_options['text'] = str(self.count)
        TextRectangle.__init__(self, canvas, x, y, width, height, options, text_options)
        MouseRectangle.regcb(self)

    def cb_left_mouse_click(self, *args, **kwargs):
        self.count = self.count + 1
        self.change_text(str(self.count))
        pass


class MDPRectangle(TextRectangle):
    NORTH = 'NORTH'
    EAST = 'EAST'
    SOUTH = 'SOUTH'
    WEST = 'WEST'
    @staticmethod
    def regcb(inst):
        pass

    def __init__(self, canvas, x, y, width, height, options={}, text_options={}, init_direction = NORTH, direction_options={}):
        self.direction = init_direction
        self.direction_options = direction_options
        self.value = 0.0
        self.dir_id = None
        TextRectangle.__init__(self, canvas, x, y, width, height, options, text_options)
        MDPRectangle.regcb(self)

    def calc_coords_with_direction(self, direction):
        tri_x0 = self.x + self.width/2
        tri_y0 = self.y
        tri_x1 = tri_x0 - 8
        tri_y1 = tri_y0 + 8
        tri_x2 = tri_x0 + 8
        tri_y2 = tri_y0 + 8
        rot_vec_x = self.x + self.width/2
        rot_vec_y = self.y + self.height/2

        tri_mat = np.matrix( ((tri_x0 - rot_vec_x, tri_x1 - rot_vec_x, tri_x2 - rot_vec_x), \
                              (tri_y0 - rot_vec_y, tri_y1 - rot_vec_y, tri_y2 - rot_vec_y)) )
        theta = 0.0
        if direction == MDPRectangle.NORTH:
            theta = 0.0

        if direction == MDPRectangle.EAST:
            theta = math.pi/2.0

        if direction == MDPRectangle.SOUTH:
            theta = math.pi

        if direction == MDPRectangle.WEST:
            theta = 3.0 * math.pi /2.0

        rot_mat = np.matrix((\
            (math.cos(theta), -math.sin(theta)), \
            (math.sin(theta), math.cos(theta))\
        ))


        tri_mat = np.dot(rot_mat, tri_mat)
        tri_mat[0,] = np.add(tri_mat[0,], rot_vec_x)
        tri_mat[1,] = np.add(tri_mat[1,], rot_vec_y)

        convert = np.array(tri_mat.reshape(tri_mat.size, order='F')).flatten()

        return tuple(convert)
        pass

    def draw(self):
        TextRectangle.draw(self)
        self.change_text(str(self.value))
        #self.draw_direction()

    def draw_direction(self):
        t = self.calc_coords_with_direction(self.direction)
        self.dir_id = self.canvas.create_polygon(t, self.direction_options)

    def change_color(self, color):
        self.canvas.itemconfigure(self.obj_id, fill=color)

    def change_value(self, value, sign=False):
        s = ''
        if value is not None:
            self.value = value
            s = str(self.value)

        if sign:
            s = '+' + s

        self.canvas.itemconfigure(self.text_id, text = s)

    def change_direction(self, direction):
        if self.dir_id != None:
            self.canvas.delete(self.dir_id)

        self.direction = direction
        self.draw_direction()
