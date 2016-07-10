from Tkinter import*
from grid import MDPGrid
from rectangle import MDPRectangle
import tkFont

WIDTH = 500
HEIGHT = 400
OFFSET_X = 70
OFFSET_Y = 30
REC_WIDTH = 90
REC_HEIGHT = 90
root = Tk()
canvas = Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack(expand=YES, fill=BOTH)


def left_mouse_click(event):
    n_grid.dispatch('left-mouse-click', event=event)
    pass


def right_mouse_click(event):
    n_grid.dispatch('right-mouse-click', event=event, \
                    text_options={'fill':'white'},\
                    init_direction = MDPRectangle.WEST, \
                    direction_options={'fill':'white'})


def mouse_move(event):
    canvas.itemconfig(x_pos, text=str(event.x))
    canvas.itemconfig(y_pos, text=str(event.y))
    pass

x_pos = canvas.create_text(20, HEIGHT-20, text="0", fill="white")
y_pos = canvas.create_text(60, HEIGHT-20, text="0", fill="white")
times_font = tkFont.Font(family='Times', size=-20, weight='bold')
text_iter = canvas.create_text(\
    OFFSET_X + 2*REC_WIDTH-50, \
    OFFSET_Y + 20 + 3*REC_HEIGHT,\
    text="Value Iterations:",\
    fill="yellow", font=times_font)

num_iter = canvas.create_text(\
    OFFSET_X + 2*REC_WIDTH+50,\
    OFFSET_Y + 20 + 3*REC_HEIGHT,\
    text="0", fill="yellow", font=times_font)

#canvas.create_polygon((100, 100, 100, 110, 110, 100), fill = "red")
canvas.bind("<Motion>", mouse_move)
canvas.bind("<Button-1>", left_mouse_click)
canvas.bind("<Button-3>", right_mouse_click)

settings = {'fill':'black','outline': 'white'}

n_grid = MDPGrid(\
    canvas, x=OFFSET_X, y=OFFSET_Y,\
    num_x=4, num_y=3, rec_width=REC_WIDTH,\
    rec_height=REC_HEIGHT, rec_options = settings,\
    num_iter_id = num_iter, text_iter = text_iter)

n_grid.draw(text_options={'fill':'white'},init_direction = MDPRectangle.WEST, direction_options={'fill':'white'})

# config for 4x3 world to run MDP algorithms
n_grid.config(discount_factor=1, action_reward=-0.04, mode=MDPGrid.POLICY_ITER)
n_grid.set_start_point(0, 2, 'blue')
n_grid.set_goal_reward(3, 0, 1, 'green')
n_grid.set_pit_reward(3, 1, -1, 'red')
n_grid.set_wall(1,1, 'gray')


# run

#root.attributes("-toolwindow", 1)
root.resizable(0,0)
root.mainloop()