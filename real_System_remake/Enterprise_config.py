def default_actWNDF(enterprise: any, target: str, action: any):  # action: -0.5~0.5
    res = enterprise.__dict__['money'] * (action + 0.5)
    return res


def default_act_shop(enterprise: any, target: str, action: any):  # action: -0.5~0.5

    if enterprise.__dict__['intention_policy'][target] == 0:
        return 10 * (action + 0.5)
    return enterprise.__dict__['intention_policy'][target] * (1 + action)

    # 因为price决策是上一回合给的，此处的next_price是上回合赋值的，等价于这回合的price
    # 所以将此刻next_price的值赋给price，并修改next_price的值以供下回合使用


def default_actPrice(enterprise: any, target: str, action: any):  # action: -0.5~0.5
    res = enterprise.__dict__['next_price']
    enterprise.__dict__['next_price'] = enterprise.__dict__['next_price'] * (1 + action)
    return res

class Enterprise_config:

    def __init__(self,
                 name: str,
                 output_name: str,
                 money: float = 0.0,      # 初始资金
                 WNDF: float = 100.0,     # 第一回合默认借贷额度
                 stock: float = 10.0,       # 初始库存
                 price: float = 10.0,        # 初始价格
                 intention: float = 0.0,    # 初始购买意愿
                 gamma: float = 0.95,            # 每日统计数据如利润为 总利润 = γ * 总利润 + (1-γ) * 今日利润
                 action_function:dict = None,      # 类型为字典 str:function()，决定设置各变量的方法
                 ):
        self.name = name
        self.output_name = output_name
        self.money = money
        self.WNDF = WNDF
        self.stock = stock
        self.price = price
        self.intention = intention
        self.gamma = gamma
        self.action_function = action_function
        if self.action_function is None:
            self.action_function = {
                                    'WNDF':default_actWNDF,
                                    'K':default_act_shop,
                                    'L':default_act_shop,
                                    'price':default_actPrice
                                    }









