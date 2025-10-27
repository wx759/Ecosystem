

class Order:
    def __init__(self, buyer: str, seller: str, shop: str, price: float, num: float):
        self.buyer = buyer
        self.seller = seller
        self.shop = shop
        self.price = price
        self.num = num




    def __str__(self):
        res = str(self.buyer) + ' 从 ' + str(self.seller) \
              + ' 以 ' + str(self.price) + ' 的价格买了 ' +\
              str(self.num) + ' 个 ' + str(self.shop)
        # res = 'buyer:' + str(self.buyer) + ' seller:' + str(self.seller) + \
        #      ' shop:' + str(self.shop) + ' price:' + str(self.price) + " num:" + str(self.num)
        return res
