def default_actWNDB(bank: any, target: str, action: any):  # action:0~1
    res = bank.__dict__['observation'][target].WNDF * (action+0.5)
    if bank.observation[target].WNDF == 100.0:
        res = 100.0
    return res


class Bank_config:
    def __init__(self,
                 name: str,
                 fund: float = 100,
                 fund_rate: float = 1,
                 fund_increase: float = 0.1,
                 debt_time: int = 5,
                 debt_i: float = 0.005,
                 action_function: dict = None,  # 类型为字典 str:function()，决定设置各变量的方法
                 ):
        self.name = name
        self.fund = fund
        self.fund_rate = fund_rate
        self.fund_increase = fund_increase
        self.debt_time = debt_time
        self.debt_i = debt_i
        self.action_function = action_function
        if self.action_function is None:
            self.action_function = {
                                    'WNDB':default_actWNDB,
                                    }








