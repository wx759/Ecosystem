Enterprise: 企业组件

企业状态空间
state = [
0            self.money/1000,           # 1  企业当前现金 #1000
1            self.stock/100,           # 2  企业当前存货 #100
2            self.debt/1000,            # 3  企业当前欠款总额 #1000
3            self.sales/10,         # 4  企业上回合售出
4            self.output/10,            # 5  企业上回合产出
5            self.price/10,           # 8  企业上回合定价(上回合的P) #10
6            self.next_price/10,      # 9  企业这回合定价(上回合的NP) #10
7            self.WNDF/1000,            # 10  企业上回合决策的借贷金额 #1000
8            self.get_WNDF/1000,        # 11 企业上回合实际拿到的借贷金额 #1000
9            self.should_payback/1000,  # 12 企业本回合待偿还本金 #1000
10            self.iDebt/10             # 13 企业本回合待偿还利息 #10

11            self.intention_policy[K]/10  # 14 16企业上回合K/L需求数量(原始数据而非执行端重新分配后)
12            self.intention_policy[L]/10

            企业购买记录 【buyer】【seller】
13            self.market_record[production1][production1]['price']/10 
14            self.market_record[production1][production1]['num']/10 
15            self.market_record[production1][consumpution1]['price']/10 
16            self.market_record[production1][consumpution1]['num']/10 
17            self.market_record[production1][production1_thirdmarket]['price']/10 
18            self.market_record[production1][production1_thirdmarket]['num']/10
19            self.market_record[production1][consumption1_thirdmarket]['price']/10 
20            self.market_record[production1][consumption1_thirdmarket]['num']/10
            
21            self.market_record[consumption1][production1]['price']/10 
22            self.market_record[consumption1][production1]['num']/10 
23            self.market_record[consumption1][consumpution1]['price']/10 
24            self.market_record[consumption1][consumpution1]['num']/10 
25            self.market_record[consumption1][production1_thirdmarket]['price']/10 
26            self.market_record[consumption1][production1_thirdmarket]['num']/10
27            self.market_record[consumption1][consumption1_thirdmarket]['price']/10 
28            self.market_record[consumption1][consumption1_thirdmarket]['num']/10
            企业商品价格
29            self.val[K]/10 # K=10
30            self.val[K]/10 # K=100
31            self.val[L]/10 # L=10
32            self.val[L]/10 # L=100
        ]