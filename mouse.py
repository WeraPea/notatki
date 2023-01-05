#!/usr/bin/env python
from pynput.mouse import Button, Controller
from pynput import keyboard
from threading import Thread
import time
import random

a = True
b = True


def mouse_move():
    global a
    global b
    mouse = Controller()

    while True:
        time.sleep(1)
        print(a)
        if a:
            mouse.move(1, 1)
            if random.randint(1, 5) == 1:
                if b:
                    mouse.press(Button.left)
                else:
                    mouse.release(Button.left)
                b = not b


def keyboard_keys():
    def on_activate():
        global a
        a = not a

    def for_canonical(f):
        return lambda k: f(l.canonical(k))

    hotkey = keyboard.HotKey(
        keyboard.HotKey.parse('<ctrl>+<alt>+<shift>+q'),
        on_activate)
    with keyboard.Listener(
            on_press=for_canonical(hotkey.press),
            on_release=for_canonical(hotkey.release)) as l:
        l.join()


Thread(target=mouse_move).start()
Thread(target=keyboard_keys).start()


# Press and release

# mouse.press(Button.left)
# mouse.release(Button.left)

# # Double click; this is different from pressing and releasing
# # twice on macOS
# mouse.click(Button.left, 2)
