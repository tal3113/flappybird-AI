import pygame
import neat
import time
import os
import random


gen = 0

pygame.init()
# ------------------- Set Window ------------------- #
WIDTH, HEIGHT = 400, 650
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))


# ------------------- Set Caption ------------------- #
pygame.display.set_caption("Flappy Bird")


# ------------------- Load Images ------------------ #
BIRD_IMGS = [
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("imgs", "bird3.png")))
]
PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale(pygame.image.load(
    os.path.join("imgs", "bg.png")), (WIDTH, HEIGHT + 160))


# ------------------- Constants -------------------- #
FPS = 30


# --------------------- Colors --------------------- #
WHITE = (255, 255, 255)


# --------------------- Fonts ---------------------- #
STAT_FONT = pygame.font.SysFont("conicsans", 35)


# --------------------- Classes -------------------- #
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROTATION_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        # how many fixels to move up or down in this frame - displacment
        # creating an arc for the bird
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2

        # move the bird
        self.y += d

        # tilting the bird
        if d < 0 or self.y < self.height + 50:  # tilting upwards
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:                                   # tilting downwards
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VEL

    def draw(self):
        self.img_count += 1
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        WINDOW.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5
    PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
    PIPE_BOTTOM = PIPE_IMG

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 350)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self):
        WINDOW.blit(self.PIPE_TOP, (self.x, self.top))
        WINDOW.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self):
        WINDOW.blit(self.IMG, (self.x1, self.y))
        WINDOW.blit(self.IMG, (self.x2, self.y))


class Display:
    def __init__(self, genomes, config):
        self.base = Base(580)
        self.pipes = [Pipe(500)]
        self.birds = []
        self.ge = []
        self.nets = []
        self.score = 0
        self.pipe_ind = 0
        self.rem = []
        self.add_pipe = False
        self.initialize(genomes, config)

    def initialize(self, genomes, config):
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            self.nets.append(net)
            self.birds.append(Bird(170, 250))
            g.fitness = 0
            self.ge.append(g)

    def draw_window(self, gen):
        WINDOW.blit(BG_IMG, (0, 0))
        text_score = STAT_FONT.render(f"Score: {self.score}", 1, WHITE)
        WINDOW.blit(text_score, (WIDTH - text_score.get_width() - 10, 10))
        text_gen = STAT_FONT.render(f"Gen: {gen}", 1, WHITE)
        WINDOW.blit(text_gen, (10, 10))
        for pipe in self.pipes:
            pipe.draw()
        for bird in self.birds:
            bird.draw()
        self.base.draw()
        pygame.display.update()

    def moving_the_birds(self):
        for i, bird in enumerate(self.birds):
            bird.move()
            self.ge[i].fitness += 0.1
            output = self.nets[i].activate((
                bird.y,
                abs(bird.y - self.pipes[self.pipe_ind].height),
                abs(bird.y - self.pipes[self.pipe_ind].bottom)
            ))
            if output[0] > 0.5:
                bird.jump()

    def bird_hitting_ground_or_sky(self):
        for i, bird in enumerate(self.birds):
            if bird.y + bird.img.get_height() >= 580 or bird.y < 0:
                self.birds.pop(i)
                self.nets.pop(i)
                self.ge.pop(i)

    def adding_pipe(self):
        self.score += 1
        for g in self.ge:
            g.fitness += 5
        self.pipes.append(Pipe(550))
        # return score

    def handle_collision(self, i):
        self.ge[i].fitness -= 1
        self.birds.pop(i)
        self.nets.pop(i)
        self.ge.pop(i)

    def pipe_interection_bird(self, pipe, i, bird, add_pipe):
        if pipe.collide(bird):
            self.handle_collision(i)

        if not pipe.passed and pipe.x < bird.x:
            pipe.passed = True
            add_pipe = True
        return add_pipe

    def handle_movement_pipes(self, rem):
        add_pipe = False
        for pipe in self.pipes:
            for i, bird in enumerate(self.birds):
                add_pipe = self.pipe_interection_bird(pipe, i ,bird, add_pipe)

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()
        return add_pipe

    def handle_pipes(self):
        rem = []
        add_pipe = self.handle_movement_pipes(rem)

        if add_pipe:
            self.adding_pipe()

        for r in rem:
            self.pipes.remove(r)

    def there_are_birds(self):
        self.pipe_ind = 0
        if len(self.birds) > 0:
            if len(self.pipes) > 1 and self.birds[0].x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
                self.pipe_ind = 1
            return True
        else:
            return False


# ------------------- Functions -------------------- #

def main(genomes, config):
    global gen
    gen += 1
    display = Display(genomes, config)
    run = True
    clock = pygame.time.Clock()

    while run:

        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if not display.there_are_birds():
            run = False
            break

        display.moving_the_birds()
        display.handle_pipes()
        display.bird_hitting_ground_or_sky()

        display.base.move()
        display.draw_window(gen)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feed.txt")
    run(config_path)
