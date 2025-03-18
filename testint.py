from datetime import datetime
import os

class Singleton:
    _instance = None
    _folder_name = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d%H%M%S")
            folder_name = f"./output/record{formatted_time}"
            os.makedirs(folder_name)
            cls._folder_name = folder_name
        return cls._instance
    def get_folder(self):
        return self._folder_name


# 测试代码
if __name__ == "__main__":
    # 第一次调用
    s1 = Singleton()
    print(f"First instance created at: {s1.get_folder()}")

    # 第二次调用
    s2 = Singleton()
    print(f"Second instance returns first call time: {s2.get_folder()}")

    # 验证是否是同一个实例
    print(f"Are s1 and s2 the same instance? {s1 is s2}")