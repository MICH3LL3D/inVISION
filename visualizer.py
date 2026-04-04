import math
import pygame

"""
usage dictionary:
-- UP, DOWN, LEFT, RIGHT, q, e - rotates cube
-- w - zoom in
-- s - zoom out
"""

WIDTH, HEIGHT = 1000, 700
BG = (20, 20, 30)
WHITE = (240, 240, 240)
BLUE = (100, 200, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Wireframe Viewer")
clock = pygame.time.Clock()

# 3D math is hell on earth
def rotate_x(point, angle):
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y2 = y * cos_a - z * sin_a
    z2 = y * sin_a + z * cos_a
    return (x, y2, z2)

def rotate_y(point, angle):
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a + z * sin_a
    z2 = -x * sin_a + z * cos_a
    return (x2, y, z2)

def rotate_z(point, angle):
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a - y * sin_a
    y2 = x * sin_a + y * cos_a
    return (x2, y2, z)
    
def project(point, fov, viewer_distance):
    x, y, z = point

    # Prevent divide-by-zero when too close
    z += viewer_distance
    if z <= 0.1:
        z = 0.1

    factor = fov / z
    screen_x = x * factor + WIDTH / 2
    screen_y = -y * factor + HEIGHT / 2
    return (int(screen_x), int(screen_y))

# Model cube
vertices = [
    (-1, -1, -1),
    ( 1, -1, -1),
    ( 1,  1, -1),
    (-1,  1, -1),
    (-1, -1,  1),
    ( 1, -1,  1),
    ( 1,  1,  1),
    (-1,  1,  1),
]
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

scale = 120
angle_x = 0
angle_y = 0
angle_z = 0
fov = 400
viewer_distance = 5

running = True
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Rotate object with arrow keys
    if keys[pygame.K_LEFT]:
        angle_y -= 1.5 * dt
    if keys[pygame.K_RIGHT]:
        angle_y += 1.5 * dt
    if keys[pygame.K_UP]:
        angle_x -= 1.5 * dt
    if keys[pygame.K_DOWN]:
        angle_x += 1.5 * dt
    if keys[pygame.K_q]:
        angle_z -= 1.5 * dt
    if keys[pygame.K_e]:
        angle_z += 1.5 * dt

    # Zoom in/out by changing viewer distance
    if keys[pygame.K_w]:
        viewer_distance -= 2.0 * dt
    if keys[pygame.K_s]:
        viewer_distance += 2.0 * dt

    screen.fill(BG)

    transformed = []
    for vertex in vertices:
        x, y, z = vertex

        point = (x * scale, y * scale, z * scale)

        # Apply rotations
        point = rotate_x(point, angle_x)
        point = rotate_y(point, angle_y)
        point = rotate_z(point, angle_z)

        transformed.append(point)

    projected = [project(p, fov, viewer_distance * scale) for p in transformed]

    for start, end in edges:
        pygame.draw.line(screen, WHITE, projected[start], projected[end], 2)

    for px, py in projected:
        pygame.draw.circle(screen, BLUE, (px, py), 5)

    pygame.display.flip()

pygame.quit()