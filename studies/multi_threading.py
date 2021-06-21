# multithreading library
# import threading
#
# print('hello main')
# print()
#
# # method based
# def display(str, str1):
#     print(str, threading.currentThread().getName())
#     print(str, str1)
#
#
# t = threading.Thread(target=display, args=(input('enter first'), input('another input')))
#
# # method to run a thread
# t.start()
#
# display('hello', 'World')


# class/subclass based
from threading import *


class MyThread(Thread):
    def display(self, str):
        print('display', currentThread().getName(), end='\t')
        print(str)
    def show(self):
        print('show run by ', currentThread().getName(), end='\t')
