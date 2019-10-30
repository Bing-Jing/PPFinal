from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
from pynput import keyboard, mouse
from pynput.keyboard import Key
from pynput.keyboard import Controller as KeyBoardController
import threading

moc = MouseController()
keyc = KeyBoardController()

def moveClick(x, y, btn):
    moc.position = (x,y)
    moc.press(btn)
    moc.release(btn)

def doubleClick(x,y,btn,num=2):
    moc.position = (x,y)
    moc.click(btn,num)
def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

def mouseListener():
    def on_click(x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False

    def on_scroll(x, y, dx, dy):
        print('Scrolled {0} at {1}'.format(
            'down' if dy < 0 else 'up',
            (x, y)))

    # Collect events until released
    with mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll) as listener:
        listener.join()

    # ...or, in a non-blocking fashion:
    listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll)
    listener.start()
def keyListener():
    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    def on_release(key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

if __name__ == "__main__":
    # print("origin position:{}".format(moc.position))
    # moveClick(100,200,Button.left)
    # print("after moving:{}".format(moc.position))

    ###################################
    # Type two upper case As
    #keyc.press('A')
    #keyc.release('A')
    #with keyc.pressed(Key.shift):
    #    keyc.press('a')
    #    keyc.release('a')
    #
    ## Type 'Hello World' using the shortcut type method
    keyc.type('Hello World')

    # t = threading.Thread(target = keyListener)
    # t.start()
    # mouseListener()
    # t.join()
