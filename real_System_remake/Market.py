
from .Order import Order
from .Enterprise import Enterprise

class Market:

    def __init__(self,
                 name: str):
        self.name = name
        self.supply = {}                # 供给数量 (商品名str,dict('price':dict(卖家名,价格),'num':dict(卖家名，数量 ) ))
        self.requirement = {}           # 需求数量 (商品名str,dict(买家名str,数量float))
        self.order_buyer = {}          # 成交订单 (买家名str, list[Order]) Order会同时存入买家和卖家字典
        self.order_seller = {}           # 因为不能同一天内用卖出去赚来的钱继续买新产品，所以分为买方订单和卖方订单,以此分别处理两种类型的订单
        self.order = []
        self.subscribe_list = []        # 订阅者名单，用于完成每一笔交易时向订阅者推送订单信息

    def subscribe(self, enterprise: Enterprise):
        self.subscribe_list.append(enterprise)

    def new_day(self):
        self.supply = {}  # 供给数量 (商品名str,dict('price':dict(卖家名,价格),'num':dict(卖家名，数量 ) ))
        self.requirement = {}  # 需求数量 (商品名str,dict(买家名str,数量float))
        self.order_buyer = {}  # 成交订单 (买家名str, list[Order]) Order会同时存入买家和卖家字典
        self.order_seller = {}  # 因为不能同一天内用卖出去赚来的钱继续买新产品，所以分为买方订单和卖方订单,以此分别处理两种类型的订单
        self.order = []

    def receive(self, name: str, shop_name: str, num: float, price: float):
        if shop_name not in self.supply:
            self.supply[shop_name] = {'num': {}, 'price': {}}
        self.supply[shop_name]['num'][name] = num
        self.supply[shop_name]['price'][name] = price

    def get_require(self, name: str, require_list: dict):      # require_list dict{商品名str:数量float}
        for shop_name in require_list.keys():
            if shop_name not in self.requirement:
                self.requirement[shop_name] = {}
            self.requirement[shop_name][name] = require_list[shop_name]

    def answer_all_price(self, shop_name: str):
        try:
            return {'price': list(self.supply[shop_name]['price'].values()), 'num': list(self.supply[shop_name]['num'].values())}
        except KeyError:
            return None

    def answer_min_price(self,shop_name: str) -> float:
        try:
            return self.supply[shop_name]['price'][min(self.supply[shop_name]['price'],key=lambda x:self.supply[shop_name]['price'][x])]
        except KeyError:
            return None

    def assign(self):  # 市场根据需求供给开始分配，
        # 逐个满足需求，只会处理价格最便宜的卖家商品，卖完后移除卖家，仍有需求未得到满足则需要再次调用该方法
        for shop_key in self.requirement.keys():
            seller = min(self.supply[shop_key]['price'],key=lambda x:self.supply[shop_key]['price'][x])
            total_require = sum(self.requirement[shop_key].values())
            # 若一次性满足所有人需求，则无需再用到此项，下一回合有新的供给；如果无法满足，下次要选择价格次便宜的商品，无论哪种情况都要移除该数据
            total_offer = self.supply[shop_key]['num'].pop(seller)
            price = self.supply[shop_key]['price'].pop(seller)
            percent = min(round(total_offer/(total_require + 1e-2), 2), 1)

            for buyer in list(self.requirement[shop_key]):
                if buyer not in self.order_buyer:
                    self.order_buyer[buyer] = []
                if seller not in self.order_seller:
                    self.order_seller[seller] = []
                get_num = round(self.requirement[shop_key][buyer] * percent, 2)
                self.requirement[shop_key][buyer] = round(self.requirement[shop_key][buyer] - get_num , 2)
                if self.requirement[shop_key][buyer] < 1e-1:  # 防止round精度影响
                    del self.requirement[shop_key][buyer]     # 需求满足则删除
                order = Order(buyer=buyer, seller=seller, shop=shop_key, price=price,num=get_num)
                self.order_buyer[buyer].append(order)
                self.order_seller[seller].append(order)
                self.order.append(order)
                for enterprise in self.subscribe_list:
                    # 观察者模式。给订阅企业推送事件
                    enterprise.receive_market_record(order)


    def get_bill(self, buyer: str) -> list:  # list[Order]   # 获取账单，付钱
        try:
            return self.order_buyer.pop(buyer)
        except KeyError:
            return []

    def get_payback(self, seller: str):                     # 获取回款
        try:
            return self.order_seller.pop(seller)
        except KeyError:
            return []

    def show_order(self):
        res = "订单详情:" +'\n'
        for order in self.order:
            res += str(order) + ':\n'
        return res


    def __str__(self):
        res = self.name + \
               " \n供给:" + str(self.supply) + '\n' +\
               "需求:" + str(self.requirement) + '\n'

        return res
