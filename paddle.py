import turtle as t
import time

class Paddle():
    def __init__(self):
        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0
        self.mosthits = 0
        self.roundhits = 0

        # Setup Background
        self.win = t.Screen();
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle
        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball
        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 250)
        self.ball.dx = 3
        self.ball.dy = -3
        
        # Score
        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {} Missed: {} MH: {}".format(self.hit, self.miss, self.mosthits), align='center', font=('Courier', 20, 'normal'))

        # Events
        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle Movement
    def paddle_right(self):
        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x + 20)

    def paddle_left(self):
        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x - 20)
    def run_frame(self):
        self.win.update()

        # Ball Moving
        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)
        self.reward += 0.0001

        # Ball and Wall Collision
        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground Contact 
        if self.ball.ycor() < -290:
            if(self.mosthits < self.roundhits):
                self.mosthits = self.roundhits
                self.reward += 5

            self.roundhits = 0
            self.ball.goto(0, 250)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {} Missed: {} MH: {}".format(self.hit, self.miss, self.mosthits), align='center', font=('Courier', 20, 'normal'))
            self.reward -= 1
            self.done = True

        # Ball Paddle Contact
        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1 
            self.hit += 1
            self.roundhits += 1
            self.score.clear()
            self.score.write("Hit: {} Missed: {} MH: {}".format(self.hit, self.miss, self.mosthits), align='center', font=('Courier', 20, 'normal'))
            self.reward += 1
            if self.roundhits > self.mosthits:
                self.reward += 1

    # AI Control 
    # 0 do nothing 
    # 1 go left
    # 2 go right
    def step(self, action):
        self.reward = 0
        self.done = 0

        if action == 1:
            self.paddle_left()
            self.reward -= 0.001

        if action == 2:
            self.paddle_right()
            self.reward -= 0.001

        self.run_frame()

        state = [self.paddle.xcor() * .01, self.ball.xcor() * 0.1, self.ball.ycor() * .01, self.ball.dx, self.ball.dy]

        return self.reward, state, self.done


    def reset(self):
        self.paddle.goto(0, -275)
        self.ball.goto(0, 250)
        self.ball.dx = 3
        self.ball.dy = -3
        return [self.paddle.xcor() * .01, self.ball.xcor() * 0.1, self.ball.ycor() * .01, self.ball.dx, self.ball.dy]


# env = Paddle()

# while True:
#     env.run_frame()
#     time.sleep(.01)