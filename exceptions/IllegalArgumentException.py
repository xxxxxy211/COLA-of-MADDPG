# IllegalArgumentException 2023/10/22 20:25

class IllegalArgumentException(Exception):
    def __init__(self, arg_name, illegal_value, info):
        self.arg_name = arg_name
        self.illegal_value = illegal_value
        self.info = info

    def __str__(self):
        print("IllegalArgumentException: arg: {} get an illegal value {}. {}".format(self.arg_name,
                                                                                     self.illegal_value,
                                                                                     self.info))
