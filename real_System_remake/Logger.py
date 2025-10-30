from .Enterprise import Enterprise
from .Bank import Bank
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import shutil
from pandas import DataFrame
import swanlab as wandb


plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
base_path_default =  'D:\\run'
# OMPerror15
#KMP_DUPLICATE_LIB_OK=True

class Logger:
    def __init__(self, name,base_path:str = None):
        self.name = name
        if base_path is None:
            base_path = base_path_default
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            os.makedirs(base_path + '\\temp')

        self.save_path = base_path + "\\"+ self.name
        self.txt_path = base_path + '\\temp\\log_' + self.name + '.txt'
        self.txt_file = open(self.txt_path, 'w')
        self.config_path = base_path + '\\temp\\config_' + self.name + '.txt'
        self.config_file = open(self.config_path, 'w')
        # self.start_time = time.clock()
        self.start_time = time.perf_counter()
        # dict{主体名称str:dict{智能体名称str:dict{网络层名称str:{'0':[],'1':[],..,'111':[]}}}}
        self.network_show = {}
        # dict{主体名称str:dict{属性名称str:属性详情[]}}
        self.data = {}   # self.data['enterprise' or 'bank']['finish' or 'runtime']
        self.data['enterprise'] = {}
        self.data['enterprise']['finish'] = {}       # 用于显示回合结束数据 运行结束显示用
        self.data['enterprise']['runtime'] = {}      # 用于记录运行时数据 存入csv用
        self.data['bank'] = {}
        self.data['bank']['finish'] = {}  # 用于显示回合结束数据 运行结束显示用
        self.data['bank']['runtime'] = {}  # 用于记录运行时数据 存入csv用
        self.loss = {} # 用于记录loss
        self.action_data = {}
        # 主体名称翻译
        self.target_translator = {'production1': '生产企业1', 'consumption1': '消费企业1','production2': '生产企业2', 'consumption2': '消费企业2',
                                  'bank1': '银行', 'episode': '回合', 'day':'天数',
                                  'money': '现金',
                                  'stock': '存货', 'debt':'债务', 'revenue': '收入', 'iDebt':'利息', 'cost': '支出', 'business_profit': '商业利润',
                                  'price': '今日定价','profit':'利润', 'economy_profit':'金融利润', 'next_price':'次日定价', 'WNDF':'决策贷款意愿', 'get_WNDF': '获得贷款',
                                  'total_profit': '总利润', 'total_revenue': '总收入', 'total_cost': '总支出', 'output': '本回合生产',
                                  'sales': '本回合售出','total_sales':'总售出', 'reward':'奖励','intention_policy_K':'决策意愿_K',
                                  'intention_policy_L': '决策意愿_L','WNDB_production1':'借贷意愿_生产企业1','WNDB_consumption1':'借贷意愿_消费企业1',
                                  'WNDB_production2': '借贷意愿_生产企业2', 'WNDB_consumption2': '借贷意愿_消费企业2',
                                  'intention_policy':'决策意愿', 'get_shop': '获取商品数', 'able_fund':'剩余可用储备金', 'bond': '债券',
                                  'WNDB': '借贷意愿', 'real_WNDB': '实际借贷','total_reward':'累计奖励'}
        # 企业普通属性
        self.e_property = ['money', 'stock', 'debt', 'revenue', 'iDebt', 'cost', 'business_profit','economy_profit', 'price', 'next_price', 'WNDF',
                            'get_WNDF', 'total_profit', 'total_cost', 'total_revenue', 'output', 'sales','total_sales']
        # 企业字典变量属性
        self.e_dict = {'intention_policy': ['K', 'L'], 'get_shop': ['K', 'L'],'reward':['business','economy'],'loss':['business','economy'],
                       'total_reward':['business','economy']}

        self.show_e_action_detail = ['WNDF','intention_policy_K','intention_policy_L','price','next_price']
        self.show_b_action_detail = ['WNDB_production1','WNDB_consumption1','WNDB_production2','WNDB_consumption2']

        # 银行普通属性
        self.b_property = ['money', 'profit', 'able_fund','total_profit']

        # 银行字典数据
        self.b_dict = {'debt': ['production1','consumption1','production2','consumption2'],
                       'bond': ['production1','consumption1','production2','consumption2'],
                       'WNDB': ['production1','consumption1','production2','consumption2'],
                       'real_WNDB': ['production1','consumption1','production2','consumption2'],
                       'reward':['WNDB'],'loss':['WNDB'],'total_reward':['WNDB']}

        self.outer_info = ['episode', 'day']


    def receive_data(self ,episode:int, day:int, target, save_into, property_list, dict_list):
        name = self.key_encoder(target.name)
        if name not in save_into:
            save_into[name] = {}
        for key in self.outer_info:
            key = self.key_encoder(key)
            if key not in save_into[name]:
                save_into[name][key] = []
        # 插入回合日期
        save_into[name][self.key_encoder('day')].append(day)
        save_into[name][self.key_encoder('episode')].append(episode)
        # 插入普通数据
        for property_key in property_list:
            save_key = self.key_encoder(property_key)
            if save_key not in save_into[name]:
                save_into[name][save_key] = []
            save_into[name][save_key].append(target.__dict__[property_key])
        # 插入字典数据
        for dict_key in dict_list:
            dict_save_key = self.key_encoder(dict_key)
            for detail_key in dict_list[dict_key]:
                detail_save_key = self.key_encoder(detail_key)
                res_key = dict_save_key + "_" + detail_save_key
                if res_key not in save_into[name]:
                    save_into[name][res_key] = []
                try:
                    save_into[name][res_key].append(target.__dict__[dict_key][detail_key])
                except KeyError:
                    save_into[name][res_key].append(0)

    def receive_toshow(self, target,data):  # target主体名称 data 对应enterprise的toshow
        if target not in self.network_show:
            self.network_show[target] = {}
        if len(data)> 20:
            a=1
        for key in data:
            if key not in self.network_show[target]:
                self.network_show[target][key] = {}
            for network_name in data[key]:
                if network_name not in self.network_show[target][key]:
                    self.network_show[target][key][network_name] = {}
                for i in range(len(data[key][network_name])):
                    if str(i) not in self.network_show[target][key][network_name]:
                        self.network_show[target][key][network_name][str(i)] = []
                    self.network_show[target][key][network_name][str(i)].append(data[key][network_name][i])

    def receive_enterprise(self ,episode:int, day:int, target: Enterprise):
        self.receive_data(episode=episode, day=day,target=target, save_into=self.data['enterprise']['runtime'], property_list=self.e_property, dict_list=self.e_dict)

    def receive_finish_enterprise(self ,episode:int, day:int, target: Enterprise):
        self.receive_data(episode=episode, day=day,target=target, save_into=self.data['enterprise']['finish'], property_list=self.e_property, dict_list=self.e_dict)

    def receive_bank(self,episode:int, day:int, target: Bank):
        self.receive_data(episode=episode, day=day,target=target, save_into=self.data['bank']['runtime'], property_list=self.b_property, dict_list=self.b_dict)

    def receive_finish_bank(self,episode:int, day:int, target: Bank):
        self.receive_data(episode=episode, day=day,target=target, save_into=self.data['bank']['finish'], property_list=self.b_property, dict_list=self.b_dict)

    def receive_loss(self, loss:dict):
        for key in loss:
            if key not in self.loss:
                self.loss[key] = {}
            for detail in loss[key]:
                if detail not in self.loss[key]:
                    self.loss[key][detail] = []
                self.loss[key][detail].append(loss[key][detail])

    def receive_action(self, name: str, action: list, action_detail: list, episode):
        name = self.key_encoder(name)
        if name not in self.action_data:
            self.action_data[name] = {}
        for i in range(len(action_detail)):
            if action_detail[i] not in self.action_data[name]:
                self.action_data[name][action_detail[i]] = []
            self.action_data[name][action_detail[i]].append(action[i])
        # if episode % 10 == 0:
        #     print(name + ' 决策为 ' + str(action_detail) + " : " + str(action))


    def clear_runtime_data(self):
        self.runtime_data = {}

    def clear_finish_data(self):
        self.data['enterprise']['finish'] = {}
        self.data['bank']['finish'] = {}

    def show_action_hist(self, start_at:int=0,divive:int = 100,path:str = None):
        for key in self.action_data:
            for action in self.action_data[key]:
                # plt.figure(str(key) + "_" + str(action))
                plt.hist(self.action_data[key][action][start_at:],divive)
                self.output_graph(str(key) + "_" + str(action)+"决策增值",path)


    def show_action_detail_hist(self, start_at:int=0,divive:int = 100,path:str = None):
        for key in self.data['enterprise']['runtime']:
            key = self.key_encoder(key)
            for action in self.show_e_action_detail:
                action = self.key_encoder(action)
                # plt.figure(str(key) + "_" + action + ":detail")
                plt.hist(self.data['enterprise']['runtime'][key][action][start_at:],divive)
                self.output_graph(str(key) + "_" + action + "_详情",path)
        for key in self.data['bank']['runtime']:
            key = self.key_encoder(key)
            for action in self.show_b_action_detail:
                action = self.key_encoder(action)
                # plt.figure(str(key) + "_" + action + ":detail")
                plt.hist(self.data['bank']['runtime'][key][action][start_at:],divive)
                self.output_graph(str(key) + "_" + action + "_详情",path)

    def show_network(self, start_at:int=0,divive:int = 100,path:str = None):
        if not os.path.exists(path + '\\network'):
            os.makedirs(path + '\\network')
        path = path + '\\network'
        for target in self.network_show:
            for key in self.network_show[target]:
                for network_name in self.network_show[target][key]:
                    for num in self.network_show[target][key][network_name]:
                        plt.hist(self.network_show[target][key][network_name][num][start_at:],divive)
                        self.output_graph(str(target) + "_" + str(key)+"_"+str(num),path + '\\' + str(network_name))

    def show_day(self,start_at:int=0,path=None):
        day = self.data['enterprise']['finish']['消费企业1']['天数'][start_at:]
        res = [0]
        for i in range(len(day)):
            if i % 10 == 9:
                res[len(res) - 1] /= 10
                res.append(0)

            res[i // 10] += day[i]
        plt.plot(res)
        self.output_graph("每十回合平均存活天数", save_path=path, xlabel='回合/10', ylabel='平均存活天数')

        count = 0
        res = []
        for i in range(len(day)):
            count += day[i]
            if i >= 100:
                count -= day[i - 100]
                res.append(count / 100)
        plt.plot(res)
        self.output_graph("每百回合平均存活天数", save_path=path, xlabel='回合/100', ylabel='平均存活天数')


    def extra_show(self, start_at:int = 0, path:str = None):
        if path is None:
            path = self.save_path

        self.output_discount(target='消费企业1',data_name='总利润',start_at=start_at,path=path)
        self.output_discount(target='生产企业1',data_name='总利润',start_at=start_at,path=path)
        self.output_discount(target='消费企业2', data_name='总利润', start_at=start_at, path=path)
        self.output_discount(target='生产企业2', data_name='总利润', start_at=start_at, path=path)
        # self.output_discount(target='银行', data_name='现金', start_at=start_at, path=path,agent_type='bank')
        # self.output_discount(target='银行', data_name='利润', start_at=start_at, path=path,agent_type='bank')


        self.output_discount(target='消费企业1', data_name='商业利润', start_at=start_at, path=path,type='runtime')
        self.output_discount(target='生产企业1', data_name='商业利润', start_at=start_at, path=path,type='runtime')
        self.output_discount(target='消费企业2', data_name='商业利润', start_at=start_at, path=path, type='runtime')
        self.output_discount(target='生产企业2', data_name='商业利润', start_at=start_at, path=path, type='runtime')


        self.show_day(start_at=start_at,path=path)


        enterprise_reward_mul = 100
        bank_reward_mul = 100

        reward_name_list = ['累计奖励_business']
        target_name_list = ['生产企业1','消费企业1','生产企业2','消费企业2']
        # line color
        c_list = {'生产企业1':'r','消费企业1':'y','生产企业2':'b','消费企业2':'g'}
        for target in target_name_list:
            for reward_name in reward_name_list:
                try:
                    reward = self.data['enterprise']['finish'][target][reward_name][start_at:]
                    res = [0]
                    for i in range(len(reward)):
                        if i % 10 == 9:
                            res[len(res)-1] /= 10
                            res.append(0)
                        res[i//10] += (reward[i] * enterprise_reward_mul)
                    plt.plot(res)
                    self.output_graph('每十回合' + str(target) + '累计奖励', save_path=path,xlabel='回合/10',ylabel='平均累计奖励')
                except KeyError:
                    pass
        for reward_name in reward_name_list:
            a = 0
            for target in target_name_list:
                try:
                    reward = self.data['enterprise']['finish'][target][reward_name][start_at:]
                    count=0
                    res = []
                    for i in range(len(reward)):
                        count += (reward[i] * enterprise_reward_mul)
                        if i >= 100:
                            count -= (reward[i-100] * enterprise_reward_mul)
                            res.append(count/100)
                    plt.plot(res,c=c_list[target],label=str(target))

                except KeyError:
                    pass
            plt.legend()
            self.output_graph('百回合累计奖励', save_path=path,xlabel='回合/100',ylabel='平均累计奖励')


        # 平均总利润
        for target in target_name_list:
            try:
                total_profit = self.data['enterprise']['finish'][target]['总利润'][start_at:]
                count = 0
                res = []
                for i in range(len(total_profit)):
                    count += (total_profit[i])
                    if i >= 100:
                        count -= (total_profit[i - 100])
                        res.append(count / 100)
                plt.plot(res, c=c_list[target], label=str(target))
            except KeyError:
                pass
        plt.legend()
        self.output_graph('平均总利润', save_path=path, xlabel='回合/100', ylabel='平均总利润')

        # 折算总利润
        for target in target_name_list:
            try:
                total_profit = self.data['enterprise']['finish'][target]['总利润'][start_at:]
                count = total_profit[0]
                res = []
                for i in range(len(total_profit)):
                    count = count * 0.99 + total_profit[i] * 0.01
                    res.append(count)
                plt.plot(res, c=c_list[target], label=str(target))
            except KeyError:
                pass
        plt.legend()
        self.output_graph('折算总利润', save_path=path, xlabel='回合', ylabel='折算总利润')

        # 银行百回合
        reward = self.data['bank']['finish']['银行']['累计奖励_借贷意愿'][start_at:]
        count=0
        res = []
        for i in range(len(reward)):
            count += (reward[i] * bank_reward_mul)
            if i >= 100:
                count -= (reward[i-100] * bank_reward_mul)
                if count<0:
                    res.append(count/1000)
                else:
                    res.append(count/100)
        plt.plot(res,c='y',label='银行')
        plt.legend()
        self.output_graph('百回合累计奖励_银行', save_path=path,xlabel='回合/100',ylabel='平均累计奖励/1000')

    def wandb_log(self,start_at:int=0,epi = None):
        # # 每百回合生存天数
        # day = self.data['enterprise']['finish']['消费企业1']['天数'][start_at:]
        # count = 0
        start = epi-100
        end = epi
        # for i in range(start,end):
        #     count += day[i]
        # res = count/100
        # wandb.log({'每百回合/存活天数':res})

        # 每百回合累计奖励
        #   enterprise
        reward_name_list = ['累计奖励_business']
        target_name_list = ['生产企业1', '消费企业1', '生产企业2', '消费企业2']
        enterprise_reward_mul = 100
        try:
            for reward_name in reward_name_list:
                for target_name in target_name_list:
                    reward = self.data['enterprise']['finish'][target_name][reward_name][start_at:]
                    count = 0
                    for i in range(start,end):
                        count += (reward[i] * enterprise_reward_mul)
                    res = count/100
                    if target_name == '生产企业1':
                        wandb.log({'每百回合/累计奖励/生产企业1':res})
                    elif target_name == '消费企业1':
                        wandb.log({'每百回合/累计奖励/消费企业1': res})
                    elif target_name == '生产企业2':
                        wandb.log({'每百回合/累计奖励/生产企业2':res})
                    else:
                        wandb.log({'每百回合/累计奖励/消费企业2': res})
        except KeyError:
            pass
        #    bank
        bank_reward_mul = 100
        reward = self.data['bank']['finish']['银行']['累计奖励_借贷意愿'][start_at:]
        count = 0
        for i in range(start,end):
            count += (reward[i] * bank_reward_mul)
        if count < 0:
            res = count / 1000
        else:
            res = count / 100
        wandb.log({'每百回合/累计奖励/银行':res})


    def output_discount(self,target:str,data_name:str, start_at:int = 0, path:str = None, type:str = 'finish',agent_type:str = 'enterprise'):
        if path is None:
            path = self.save_path
        try: # 先确保存在该数据
            self.data[agent_type][type][target][data_name]
        except:
            return
        if isinstance(self.data[agent_type][type][target][data_name],dict):
            data_list = {}
            for detail in self.data[agent_type][type][target][data_name]:
                data_list[detail] = self.data[agent_type][type][target][data_name][detail][start_at:]
            for key in data_list:
                self.output_discount_graph(target,data_name,data_list[key],start_at,path,type,agent_type)
        else:
            data_list = self.data[agent_type][type][target][data_name][start_at:]
            self.output_discount_graph(target,data_name,data_list,start_at,path,type,agent_type)

    def output_discount_graph(self,target:str,data_name:str,data_list:list, start_at:int = 0, path:str = None, type:str = 'finish',agent_type:str = 'enterprise'):
        discount_gamma = 0.99
        res = []
        res.append(data_list[0])
        for i in range(1, len(data_list)):
            res.append(discount_gamma * res[i - 1] + (1 - discount_gamma) * data_list[i])
        plt.plot(res, c='r', label=target)
        prifix = '单回合折算' if type is 'finish' else '运行时折算'
        self.output_graph(str(target) + str(prifix) + str(data_name) + "99_1", save_path=path)

    def finish(self):
        # end_time = time.clock()
        end_time = time.perf_counter()
        print("运行时间：",(end_time-self.start_time) / 60,'分钟')
        end_struct_time = time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
        print('结束时间',end_struct_time)
        self.save_path = self.save_path + end_struct_time
        self.txt_file.close()
        self.config_file.close()
        self.remove_config_to_target()
        self.remove_txt_to_target()



    def show_finish_data(self, start_at:int = 0, path:str = None):
        if path is None:
            path = self.save_path
        for target_key in self.data['enterprise']['finish']:
            for data_key in self.data['enterprise']['finish'][target_key]:
                # plt.figure(target_key + "_" + data_key)
                plt.plot(self.data['enterprise']['finish'][target_key][data_key][start_at:])
                self.output_graph(target_key + "_" + data_key, save_path=path)

        for target_key in self.data['bank']['finish']:
            for data_key in self.data['bank']['finish'][target_key]:
                # plt.figure(target_key + "_" + data_key)
                plt.plot(self.data['bank']['finish'][target_key][data_key][start_at:])
                self.output_graph(target_key + "_" + data_key, save_path=path)
        self.extra_show(start_at=start_at, path=path)
        self.show_loss(start_at=start_at, path=path)

    def show_loss(self, start_at:int = 0, path:str = None):
        if path is None:
            path = self.save_path
        for target_key in self.loss:
            for detail in self.loss[target_key]:
                list = self.loss[target_key][detail][start_at:]
                plt.plot(list)
                self.output_graph(target_key + "_" + detail + "_original_loss", save_path=path)
                res = []
                res.append(list[0])
                for i in range(1,len(list)):
                    res.append( res[len(res)-1] + list[i] )
                plt.plot(res)
                self.output_graph(target_key + "_" + "_changing_loss", save_path=path)
                discount_res = []
                discount_res.append(list[0])
                for i in range(1, len(list)):
                    discount_res.append(discount_res[len(discount_res) - 1] * 0.99 + list[i] * 0.01)
                plt.plot(discount_res)
                self.output_graph(target_key + "_" + "_discount_loss_99_1", save_path=path)


    def show_all(self):
        # 完整的数据展示
        self.show_action_hist()
        self.show_action_detail_hist()
        self.show_finish_data()
        # self.show_network(start_at=30000,path=self.save_path)

        # 仅展示最后10000回合（稳态状态）的数据
        self.show_action_hist(start_at=-10000,path=self.save_path + "\\稳态")
        self.show_action_detail_hist(start_at=-10000,path=self.save_path + "\\稳态")
        self.show_finish_data(start_at=-1000, path=self.save_path + "\\稳态")
        # self.show_network(start_at=-10000,path=self.save_path + "\\稳态")


    def key_encoder(self, key):
        res = key
        try:
            res = self.target_translator[key]
        except KeyError:
            pass
        return res

    def output_graph(self, name, save_path:str = None,xlabel:str = '回合',ylabel:str=None):
        if save_path is None:
            save_path = self.save_path
        plt.xlabel(xlabel)
        if ylabel is None:
            ylabel = name
        plt.ylabel(name)
        figure_save_path = save_path
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)
        plt.savefig(os.path.join(figure_save_path,name))
        plt.show()
        plt.close()

    def output_to_txt(self, data):
        print(data,file=self.txt_file)

    def output_config(self, data):
        print(data,file=self.config_file)

    def remove_txt_to_target(self):
        fpath,fname=os.path.split(self.txt_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        shutil.move(self.txt_path,self.save_path + '\\' + fname)

    def remove_config_to_target(self):
        fpath, fname = os.path.split(self.config_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        shutil.move(self.config_path, self.save_path + '\\' + fname)


    def clear_action(self):
        pass
    def to_csv(self):
        data = {}
        print('正在导出运行时数据')
        for name in self.data['enterprise']['runtime']:
            for key in self.data['enterprise']['runtime'][name]:
                data[name+ "_" + key]=self.data['enterprise']['runtime'][name][key]
        for name in self.data['bank']['runtime']:
            for key in self.data['bank']['runtime'][name]:
                data[name + "_" + key] =self.data['bank']['runtime'][name][key]
        df = DataFrame(data, dtype=float)
        df.to_csv(
            self.save_path + "\\数据.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print('正在导出每回合结束数据')

        data = {}
        for name in self.data['enterprise']['finish']:
            for key in self.data['enterprise']['finish'][name]:
                data[name + "_" + key] = self.data['enterprise']['finish'][name][key]
        for name in self.data['bank']['finish']:
            for key in self.data['bank']['finish'][name]:
                data[name + "_" + key] = self.data['bank']['finish'][name][key]
        df = DataFrame(data, dtype=float)
        df.to_csv(
            self.save_path + "\\结束数据.csv",
            index=False,
            encoding='utf-8-sig'
        )


