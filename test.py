# file_name = "test.csv"
# # if file_name.endswith('csv'):
# #     print("yes")
# test = 123
# assert file_name.endswith('csv')
# assert test == None


from multiprocessing import Queue
from multiprocessing.dummy import Process
import time
import random
import numpy as np

# if __name__ == "__main__":
#     q = Queue(3)   #初始化一个Queue对象，最多可接收3条put信息
#     q.put("消息1")
#     q.put("消息2")
#     print(q.full())
#     q.put("消息3")
#     print(q.full())
    
#      # 因为消息队列已满，再 put 会报异常，第一个 try 等待 2 秒后再抛出异常，第二个 try 立刻抛出
#     try:
#         q.put("消息4", True, 2)
#     except:
#         print("队列已满，已有：%s" % q.qsize())
        
#     try:
#         q.put_nowait("消息4")
#     except:
#         print("队列已满，已有：%s" % q.qsize())
        
#     # 读取消息时，先判断消息队列是否为空，再读取
#     if not q.empty():
#         print("----从消息队列中获取消息--")
#         for i in range(q.qsize()):
#             print(q.get_nowait())

# def write_task(q):
#     if not q.full():
#         for i in range(5):
#             q.put([[random.randint(0, 1000)], [1, 1]])
#             print('我发送 ..')
            
# def read_task(q):
#     print('hi')
#     time.sleep(0.001)
#     count = 0
#     # while not q.empty():
#         # print(q.qsize())
#         # print("读取： %s" % q.get(True, 0.001))  # 等待 2 秒，如果还没有读取到任何消息，则抛出异常
#     # if not q.empty():
#     #     print(q.get(True, 2))
#     for i in range(5):
#         if not q.empty():
#           print(q.get(True, 2))  
    
    

# if __name__ == "__main__":
#     print("---父进程开始---")
#     q = Queue()   # 父进程创建Queue，并传递给子进程
#     pw = Process(target=write_task, args=(q,))
#     pr = Process(target=read_task, args=(q,))
    
#     pw.start()
#     pr.start()
    
#     print("---等待子进程结束---")
#     pw.join()
#     pr.join()
#     print("---父进程结束---")


# x = 1
# def change(a):
#     x = 2
#     x += 1
# change(x)
# print(x)  

# x = [1, 2, 3, 5]
# y = np.sum(np.array(x, np.float)) / 4
# print(y)

# print('this epoch acc is: %0.5f' % (1 / 3))
            