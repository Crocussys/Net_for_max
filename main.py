from bin.Menu import Menu
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        menu = Menu(sys.argv[1])
    else:
        menu = Menu()
    menu.start()
