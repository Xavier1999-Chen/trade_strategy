from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression
import time
import talib

class AlgoEvent:
    def __init__(self):
        # 初始化参数
        self.last_update = {}  # 每个品种的最后更新时间
        self.price_history = {}  # 每个品种的价格历史
        self.base_spread = 0.0002  # 基础价差（2 pips）
        self.order_size = 1  # 基础挂单量，降低到1
        
        # 仓位跟踪相关变量
        self.current_position = {}  # 每个品种的当前持仓量
        self.position_cost = {}  # 每个品种的持仓成本
        self.max_position = 5  # 最大持仓量限制，降低到5
        self.order_status = {}  # 订单状态跟踪
        self.position_history = {}  # 每个品种的持仓历史记录
        
        # 交易的品种列表 - 会根据输入数据动态填充
        self.symbols = set()

        # 计算Hurst频率
        self.hurst_check_interval = timedelta(hours=1)  # 每1小时计算一次Hurst值
        self.last_hurst_check = datetime(2000, 1, 1)

        # 市场状态
        self.pre_market_regime = "unknown"
        self.market_regime = "unknown"  # 市场状态: "random_walk", "trending", "mean_reverting", "unknown"
        
        # MA Cross策略参数
        self.ma_cross_active = False
        self.ma_min_fast = 5  # 最快均线周期
        self.ma_max_fast = 50  # 最慢均线周期
        self.ma_min_slow = 20
        self.ma_max_slow = 200
        self.ma_fast_period = 7  # 默认值
        self.ma_slow_period = 14  # 默认值
        self.ma_volatility_lookback = 20  # 波动率计算窗口
        self.ma_cross_interval = timedelta(hours=1)  # 默认1小时检查一次
        self.ma_min_interval = timedelta(minutes=15)  # 最小间隔
        self.ma_max_interval = timedelta(hours=4)  # 最大间隔
        self.ma_last_trade_time = {}
        self.ma_order_ref = 0
        
        # 资金管理相关变量
        self.availableBalance = 100000  # 初始资本，固定值
        self.min_risk_percentage = 0.10  # 最小资金使用比例 (10%)
        self.max_risk_percentage = 0.50  # 最大资金使用比例 (50%)
        self.price_table = {}  # 保存每个品种的价格信息
        self.notional_multipliers = {}  # 每个品种的名义乘数
        self.capital_distribution = {}  # 每个品种的资金分配
        self.total_allocated_capital = 0  # 总分配资金
        
        # 做市商策略参数
        self.mm_min_spread = 0.0001  # 最小价差 (1 pip)
        self.mm_max_spread = 0.0010  # 最大价差 (10 pips)
        self.mm_inventory_target = 0  # 理想库存水平
        self.mm_inventory_limit = 0.7  # 库存限制触发调整的比例
        self.mm_order_lifespan = 180  # 订单有效期（秒）
        self.mm_rebalance_interval = timedelta(minutes=10)  # 调整库存的间隔
        self.mm_last_rebalance = {}  # 上次调整库存的时间
        self.contractSize = {}  # 合约大小
        
        # RSI参数
        self.rsi_period = 14  # RSI计算周期
        self.rsi_overbought = 70  # RSI超买阈值
        self.rsi_oversold = 30  # RSI超卖阈值
        self.rsi_check_interval = timedelta(minutes=5)  # RSI检查间隔
        self.last_rsi_check = {}  # 上次RSI检查时间
        self.rsi_values = {}  # 存储每个品种的RSI值
        self.rsi_position_adjustment = 0.3  # RSI触发的仓位调整比例
        
        # 风险控制VaR
        self.hist_returns = {}  # 各品种历史收益率 {symbol: deque}
        self.var_threshold = 0.05  # VaR风险阈值（5%净值）
        self.var_check_interval = timedelta(hours=4)  # 每4小时检查一次
        self.last_var_check = datetime(2000, 1, 1)
        
    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        
        # 获取交易品种的合约规格
        for instrument in mEvt.get('subscribeList', []):
            try:
                contract_spec = self.evt.getContractSpec(instrument)
                self.contractSize[instrument] = contract_spec.get("contractSize", 1)
                
                # 保存每个品种的乘数信息，用于计算真实价值
                self.notional_multipliers[instrument] = self.get_notional_multiplier(instrument, contract_spec)
                
                self.initialize_symbol(instrument)
                self.evt.consoleLog(f"品种 {instrument} 合约规格: 大小={self.contractSize[instrument]}, 乘数={self.notional_multipliers[instrument]}")
            except Exception as e:
                self.evt.consoleLog(f"获取合约规格失败: {instrument}, 错误: {str(e)}")
                self.contractSize[instrument] = 1  # 默认值
                self.notional_multipliers[instrument] = 1  # 默认乘数
                
        self.evt.start()
    
    # General
    def initialize_symbol(self, symbol):
        """初始化一个新交易品种的数据结构"""
        if symbol not in self.symbols:
            self.symbols.add(symbol)
            self.last_update[symbol] = datetime(2000, 1, 1)
            self.price_history[symbol] = deque(maxlen=1000)
            self.current_position[symbol] = 0
            self.position_cost[symbol] = 0
            self.position_history[symbol] = []
            self.price_table[symbol] = 0  # 初始化价格表
            self.capital_distribution[symbol] = 0  # 初始化资金分配
            self.ma_last_trade_time[symbol] = datetime(2000, 1, 1)
            self.mm_last_rebalance[symbol] = datetime(2000, 1, 1)
            self.hist_returns[symbol] = deque(maxlen=250)  # 存储约一年的日收益率
            self.last_rsi_check[symbol] = datetime(2000, 1, 1)  # 初始化RSI检查时间
            self.rsi_values[symbol] = 50  # 初始化RSI值
            self.evt.consoleLog(f"初始化新交易品种: {symbol}")

    def get_notional_multiplier(self, instrument, contract_spec):
        """根据品种和合约规格确定适当的名义价值乘数"""
        # 为不同类型的产品设置不同的乘数
        default_multiplier = 1
        
        # 如果是指数产品，设置较小的乘数
        if 'USD' in instrument:
            if any(index in instrument for index in ['SPX', 'NS', 'DOW', 'NDQ']):
                return 0.0001  # 指数产品使用0.01%的乘数
                
        # 如果是股票产品
        if contract_spec.get('productType') == 'Stock':
            return 1.0
            
        # 如果是外汇产品
        if instrument.endswith('JPY') or 'USD' in instrument:
            return 0.01
            
        # 其他情况使用默认乘数
        return default_multiplier

    def compute_hurst(self, prices):
        """计算Hurst指数（R/S分析法）- 使用sklearn的LinearRegression"""
        if len(prices) < 200:
            return 0.5

        lags = [30, 50, 100, 150, 200]  # 使用更长的lag周期
        rescaled_ranges = []
        
        for lag in lags:
            if len(prices) <= lag:
                continue
                
            segments = len(prices) // lag
            if segments < 1:
                continue
                
            rs = []
            for i in range(segments):
                if (i+1)*lag > len(prices):
                    break
                segment = prices[i*lag : (i+1)*lag]
                mean = sum(segment) / len(segment)
                adjusted = [p - mean for p in segment]
                cumulative = [sum(adjusted[:i+1]) for i in range(len(adjusted))]
                r = max(cumulative) - min(cumulative)
                # 计算标准差
                variance = sum([(p - mean)**2 for p in segment]) / len(segment)
                s = variance**0.5
                if s > 0:
                    rs.append(r / s)
                    
            if rs:
                rescaled_ranges.append(sum(rs) / len(rs))
        
        if len(rescaled_ranges) < 2:
            return 0.5
            
        # 使用sklearn的LinearRegression计算斜率
        x = np.array([np.log(lags[i]) for i in range(len(rescaled_ranges))]).reshape(-1, 1)
        y = np.array([np.log(rr) for rr in rescaled_ranges])
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model.coef_[0]

    def determine_market_regime(self, timestamp):
        """根据多个品种的Hurst指数确定市场状态"""
        if timestamp < self.last_hurst_check + self.hurst_check_interval:
            return self.market_regime  # 直接返回缓存值
        self.last_hurst_check = timestamp  # 更新检查时间
            
        hurst_values = []
        for symbol in self.symbols:
            if len(self.price_history[symbol]) >= 100:
                hurst = self.compute_hurst(list(self.price_history[symbol]))
                hurst_values.append(hurst)
                
        if not hurst_values:
            return "unknown"
            
        avg_hurst = sum(hurst_values) / len(hurst_values)
        self.evt.consoleLog(f"时间: {timestamp}, 平均Hurst指数: {avg_hurst:.4f}")
        
        if 0.45 <= avg_hurst <= 0.55:
            return "random_walk"  # 随机游走，适合做市
        elif avg_hurst > 0.55:
            return "trending"  # 趋势市场，适合MA Cross策略
        elif avg_hurst < 0.45:
            return "mean_reverting"  # 均值回归市场，适合调整型做市
        else:
            return "unknown"

    def distribute_capital(self):
        """为每个交易品种分配资金，确保总分配在10%-50%之间"""
        active_symbols = [s for s in self.symbols if s in self.price_table and self.price_table[s] > 0]
        
        if not active_symbols:
            return
        
        res = self.evt.getAccountBalance()
        self.availableBalance = res["availableBalance"]
            
        # 重置资金分配
        self.capital_distribution = {}
        self.total_allocated_capital = 0
        
        # 先计算每个品种的名义价值
        symbol_values = {}
        for symbol in active_symbols:
            price = self.price_table[symbol]
            multiplier = self.notional_multipliers.get(symbol, 1)
            contract_size = self.contractSize.get(symbol, 1)
            value = price * contract_size * multiplier
            symbol_values[symbol] = value
            
        # 计算总价值
        total_value = sum(symbol_values.values())
        
        if total_value <= 0:
            return
            
        # 计算可分配的总资金范围
        min_allocation = self.availableBalance * self.min_risk_percentage
        max_allocation = self.availableBalance * self.max_risk_percentage
        
        # 根据价值比例初步分配资金
        for symbol in active_symbols:
            value_ratio = symbol_values[symbol] / total_value
            # 先分配最小资金比例
            initial_allocation = min_allocation * value_ratio
            self.capital_distribution[symbol] = initial_allocation 
                
        # 如果市场状态适合，增加资金分配至最大限制
        if self.market_regime in ["random_walk", "mean_reverting", "trending"]:
            # 可以增加到最大限制
            available_increase = max_allocation - min_allocation
            for symbol in active_symbols:
                value_ratio = symbol_values[symbol] / total_value
                additional = available_increase * value_ratio
                self.capital_distribution[symbol] += additional
                
        # 更新总分配资金
        self.total_allocated_capital = sum(self.capital_distribution.values())
        
        # 记录资金分配情况
        self.evt.consoleLog(f"资金分配: 总资本={self.availableBalance:.2f}, 分配资金={self.total_allocated_capital:.2f} " +
                           f"({(self.total_allocated_capital/self.availableBalance*100):.1f}%)")
        for symbol in self.capital_distribution:
            self.evt.consoleLog(f"  {symbol}: {self.capital_distribution[symbol]:.2f} " +
                               f"({(self.capital_distribution[symbol]/self.availableBalance*100):.1f}%)")

    def calculate_order_size(self, symbol, price, is_ask=False):
        """根据价格和品种特性计算安全的订单大小"""
        if price <= 0:
            return 0.01  # 最小订单量
            
        # 获取该品种的名义乘数
        multiplier = self.notional_multipliers.get(symbol, 1)
        
        # 计算单个合约的名义价值
        contract_value = price * self.contractSize.get(symbol, 1) * multiplier
        
        # 获取分配给该品种的资金
        allocated_capital = self.capital_distribution.get(symbol, 0)
        
        # 如果没有分配资金，使用最小比例
        if allocated_capital <= 0:
            allocated_capital = self.availableBalance * self.min_risk_percentage / max(1, len(self.symbols))
            
        # 计算可交易的合约数量
        max_contracts = allocated_capital / contract_value
        
        # 针对高价值合约，确保订单量不会太大
        if contract_value > allocated_capital:
            # 如果一个合约的价值已经超过可用资金，使用最小订单量
            order_size = 0.01
        else:
            # 否则，根据资金比例计算，但不超过最大持仓
            order_size = min(max_contracts, self.max_position)
            
        # 确保订单量不小于最小值且不大于最大值
        order_size = max(0.01, min(order_size, self.max_position))
        
        # 四舍五入到2位小数
        order_size = round(order_size, 2)
        
        # 记录计算过程
        self.evt.consoleLog(f"订单量计算 - 品种: {symbol}, 价格: {price:.5f}, 合约值: {contract_value:.2f}, "
                           f"乘数: {multiplier}, 分配资金: {allocated_capital:.2f}, 订单量: {order_size:.2f}")
        
        return order_size

    # MA Cross
    def adjust_ma_periods(self, symbol):
        """根据波动率动态调整MA周期"""
        if len(self.price_history[symbol]) < self.ma_volatility_lookback:
            return
            
        # 计算价格波动率
        prices = np.array(list(self.price_history[symbol]))
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 波动率越高，使用越长的均线周期(趋势性强)
        # 波动率越低，使用越短的均线周期(均值回归)
        volatility_normalized = 1 / (1 + np.exp(-10*(volatility-0.15)))  # Sigmoid函数调整
        
        # 计算动态周期
        self.ma_fast_period = int(
            self.ma_min_fast + (self.ma_max_fast - self.ma_min_fast) * volatility_normalized
        )
        self.ma_slow_period = int(
            self.ma_min_slow + (self.ma_max_slow - self.ma_min_slow) * volatility_normalized
        )
        
        # 确保慢线周期大于快线
        self.ma_slow_period = max(self.ma_slow_period, self.ma_fast_period + 5)
        
        self.evt.consoleLog(
            f"动态调整MA周期: {symbol} 快线={self.ma_fast_period}, 慢线={self.ma_slow_period}, "
            f"波动率={volatility:.4f}"
        )

    def adjust_ma_check_frequency(self, symbol):
        """根据市场波动性调整MA信号检查频率"""
        if len(self.price_history[symbol]) < 20:
            return
            
        # 计算近期价格变化率
        prices = list(self.price_history[symbol])
        recent_changes = np.abs(np.diff(prices[-20:]) / prices[-20:-1])
        avg_change = np.mean(recent_changes)
        
        # 波动性越高，检查越频繁
        freq_ratio = min(max((avg_change - 0.001) / 0.002, 0), 1)  # 假设0.1%-0.3%是正常范围
        self.ma_cross_interval = self.ma_max_interval - (
            (self.ma_max_interval - self.ma_min_interval) * freq_ratio
        )
        
        self.evt.consoleLog(
            f"调整MA检查频率: {symbol} 新间隔={self.ma_cross_interval.total_seconds()/3600:.1f}小时, "
            f"近期平均变化={avg_change*100:.2f}%"
        )

    def check_ma_cross_signal(self, symbol, timestamp):
        """检查MA交叉信号"""
        self.adjust_ma_check_frequency(symbol)
        if timestamp < self.ma_last_trade_time.get(symbol, datetime(2000,1,1)) + self.ma_cross_interval:
            return
            
        # 动态调整MA周期
        self.adjust_ma_periods(symbol)
        
        if len(self.price_history[symbol]) < self.ma_slow_period + self.ma_fast_period:
            self.evt.consoleLog(f"{symbol} 数据不足,无法计算MA")
            return
            
        # 获取历史数据
        close_prices = np.array(list(self.price_history[symbol]))
        
        # 计算MA
        fast_ma = talib.SMA(close_prices, timeperiod=self.ma_fast_period)
        slow_ma = talib.SMA(close_prices, timeperiod=self.ma_slow_period)
        
        # 检查是否有足够的数据
        if np.isnan(fast_ma[-1]) or np.isnan(fast_ma[-2]) or np.isnan(slow_ma[-1]) or np.isnan(slow_ma[-2]):
            return
            
        # 计算波动率止损
        atr = talib.ATR(
            np.array([p for p in self.price_history[symbol]]),  # high
            np.array([p for p in self.price_history[symbol]]),  # low
            close_prices,
            timeperiod=14
        )[-1]
        
        # 获取当前价格
        current_price = self.price_history[symbol][-1]
        
        # 检查金叉/死叉
        golden_cross = fast_ma[-1] > slow_ma[-1] and fast_ma[-2] <= slow_ma[-2]
        death_cross = fast_ma[-1] < slow_ma[-1] and fast_ma[-2] >= slow_ma[-2]
        
        if golden_cross or death_cross:
            # 计算订单量
            volume = self.calculate_order_size(symbol, current_price)
            volume = max(0.01, min(volume, self.max_position))
            
            # 生成订单引用
            self.ma_order_ref += 1
            order_ref = f"MA_{self.ma_order_ref}"
            
            # 基于波动率设置止损
            stop_loss_multiplier = 1  # 止损倍数
            take_profit_multiplier = 4  # 止盈倍数
            
            if golden_cross:
                # 动态止损止盈
                stop_loss = current_price - stop_loss_multiplier * atr
                take_profit = current_price + take_profit_multiplier * atr
                
                # 发送买入订单
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=order_ref,
                    volume=volume,
                    openclose='open',
                    buysell=1,
                    ordertype=0,  # 市价单
                    takeProfitLevel=take_profit,
                    stopLossLevel=stop_loss
                )
                self.evt.sendOrder(order)
                self.evt.consoleLog(
                    f"MA金叉信号: 买入 {symbol} {volume:.2f} @ {current_price:.5f}, "
                    f"止损={stop_loss:.5f}, 止盈={take_profit:.5f}, ATR={atr:.5f}"
                )
                
            elif death_cross:
                # 动态止损止盈
                stop_loss = current_price + stop_loss_multiplier * atr
                take_profit = current_price - take_profit_multiplier * atr
                
                # 发送卖出订单
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=order_ref,
                    volume=volume,
                    openclose='open',
                    buysell=-1,
                    ordertype=0,  # 市价单
                    takeProfitLevel=take_profit,
                    stopLossLevel=stop_loss
                )
                self.evt.sendOrder(order)
                self.evt.consoleLog(
                    f"MA死叉信号: 卖出 {symbol} {volume:.2f} @ {current_price:.5f}, "
                    f"止损={stop_loss:.5f}, 止盈={take_profit:.5f}, ATR={atr:.5f}"
                )
            
            # 更新最后交易时间
            self.ma_last_trade_time[symbol] = timestamp    

    def close_all_ma_cross_positions(self):
        """平仓所有MA Cross策略的持仓"""
        # 获取系统中的所有未平仓订单
        pos, osOrder, pendOrder = self.evt.getSystemOrders()
        
        # 遍历所有未平仓订单，寻找由MA Cross策略开仓的订单
        for tradeID in osOrder:
            order = osOrder[tradeID]
            # 检查订单引用是否以"MA_"开头
            if order.get('orderRef', '').startswith('MA_'):
                # 发送平仓订单（反向操作）
                close_order = AlgoAPIUtil.OrderObject(
                    instrument=order['instrument'],
                    openclose='close',
                    buysell=-1 * order['buysell'],
                    ordertype=0,  # 市价单
                    volume=order['Volume'],
                    orderRef=f"CLOSE_MA_{order['orderRef']}"
                )
                self.evt.sendOrder(close_order)
                self.evt.consoleLog(f"平仓MA Cross订单: {order['orderRef']} 数量{order['Volume']}")

        # 清空MA相关持仓记录（通过强制归零）
        for symbol in self.symbols:
            if self.current_position[symbol] != 0:
                # 发送对冲订单确保仓位清零
                hedge_volume = abs(self.current_position[symbol])
                hedge_side = -1 if self.current_position[symbol] > 0 else 1
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    openclose='close',
                    buysell=hedge_side,
                    ordertype=0,
                    volume=hedge_volume,
                    orderRef=f"FORCE_CLOSE_MA_{symbol}"
                )
                self.evt.sendOrder(order)
                self.current_position[symbol] = 0  # 强制更新本地记录

    # Market Making
    def calculate_rsi(self, symbol):
        """计算RSI指标"""
        if len(self.price_history[symbol]) < self.rsi_period + 1:
            return 50  # 数据不足时返回中性值
            
        prices = np.array(list(self.price_history[symbol]))
        deltas = np.diff(prices)
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均上涨和下跌
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100  # 避免除以零
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def adjust_position_by_rsi(self, symbol, md):
        """根据RSI调整持仓"""
        if md.timestamp < self.last_rsi_check[symbol] + self.rsi_check_interval:
            return
            
        self.last_rsi_check[symbol] = md.timestamp
        current_rsi = self.calculate_rsi(symbol)
        self.rsi_values[symbol] = current_rsi
        
        # 记录RSI值
        self.evt.consoleLog(f"RSI更新 - 品种: {symbol}, RSI: {current_rsi:.2f}")
        
        # 根据RSI值调整持仓
        if current_rsi > self.rsi_overbought and self.current_position[symbol] > 0:
            # 超买且持有多头，减仓
            reduce_volume = abs(self.current_position[symbol]) * self.rsi_position_adjustment
            reduce_volume = round(reduce_volume, 2)
            
            if reduce_volume >= 0.01:
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=f"{symbol}_RSI_SELL",
                    volume=reduce_volume,
                    openclose='close',
                    buysell=-1,
                    ordertype=0  # 市价单
                )
                self.evt.sendOrder(order)
                self.evt.consoleLog(f"RSI超买信号: 卖出 {symbol} {reduce_volume:.2f}手")
                
        elif current_rsi < self.rsi_oversold and self.current_position[symbol] < 0:
            # 超卖且持有空头，减仓
            reduce_volume = abs(self.current_position[symbol]) * self.rsi_position_adjustment
            reduce_volume = round(reduce_volume, 2)
            
            if reduce_volume >= 0.01:
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=f"{symbol}_RSI_BUY",
                    volume=reduce_volume,
                    openclose='close',
                    buysell=1,
                    ordertype=0  # 市价单
                )
                self.evt.sendOrder(order)
                self.evt.consoleLog(f"RSI超卖信号: 买入 {symbol} {reduce_volume:.2f}手")

    def place_market_making_orders(self, symbol, md):
        """核心做市逻辑，考虑当前持仓和市场状态"""
      
        # 更新资金分配
        if len(self.capital_distribution) == 0 or symbol not in self.capital_distribution:
            self.distribute_capital()
            
        # 更新仓位历史
        self.position_history[symbol].append({
            'timestamp': md.timestamp,
            'position': self.current_position[symbol],
            'mid_price': md.midPrice,
            'position_value': self.current_position[symbol] * md.midPrice
        })
        
        # 记录当前状态
        self.evt.consoleLog(f"时间: {md.timestamp}, 品种: {symbol}, 当前持仓: {self.current_position[symbol]}, " 
                           f"持仓成本: {self.position_cost[symbol]:.5f}, 中间价: {md.midPrice:.5f}")
        
        # 计算Hurst指数
        H = self.compute_hurst(list(self.price_history[symbol]))
        
        # 动态调整参数基于市场状态
        base_spread = self.base_spread
        # 均值回归市场 - 收窄价差
        if self.market_regime == "mean_reverting":
            base_spread = max(self.mm_min_spread, self.base_spread * 0.7)
            self.evt.consoleLog(f"均值回归市场 - 收窄价差至 {base_spread:.6f}")
            
            # 在均值回归市场中使用RSI调整持仓
            self.adjust_position_by_rsi(symbol, md)
            
        # 趋势市场 - 扩大价差
        elif self.market_regime == "trending":
            base_spread = min(self.mm_max_spread, self.base_spread * 1.5)
            self.evt.consoleLog(f"趋势市场 - 扩大价差至 {base_spread:.6f}")
        
        # 根据Hurst指数微调价差
        spread_multiplier = 1.0
        if H < 0.4:
            spread_multiplier = 0.8  # 强均值回归，缩小价差
        elif H > 0.6:
            spread_multiplier = 1.5  # 强趋势，扩大价差
            
        # 根据RSI值调整价差
        current_rsi = self.rsi_values.get(symbol, 50)
        if current_rsi > self.rsi_overbought or current_rsi < self.rsi_oversold:
            spread_multiplier *= 1.2  # 在RSI极值区域扩大价差
            
        # 生成订单引用标识（每个品种独立）
        bid_ref = f"{symbol}_BID"
        ask_ref = f"{symbol}_ASK"
        
        # 获取所有未成交的挂单
        _, _, pending_orders = self.evt.getSystemOrders()

        # 遍历所有挂单，取消当前品种的做市订单
        for order_id in list(pending_orders.keys()):
            order = pending_orders[order_id]
            # 检查订单是否属于当前品种且是BID/ASK做市订单
            if (
                order["instrument"] == symbol 
                and (order["orderRef"].startswith(f"{symbol}_BID") 
                     or order["orderRef"].startswith(f"{symbol}_ASK"))
            ):
                # 发送取消订单请求
                cancel_order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=order["orderRef"],
                    openclose='close'  # 关闭指定订单
                )
                self.evt.sendOrder(cancel_order)
                
                # 清除本地订单状态记录
                if order["orderRef"] in self.order_status:
                    del self.order_status[order["orderRef"]]
                
                self.evt.consoleLog(f"取消残留做市订单: {order['orderRef']}")
        
        # 计算库存偏移量 (与目标库存的差距)
        inventory_deviation = self.current_position[symbol] - self.mm_inventory_target
        inventory_ratio = abs(inventory_deviation) / self.max_position if self.max_position > 0 else 0
        
        # 计算动态价差调整因子 - 库存越偏离目标，调整越大
        skew_factor = min(0.8, inventory_ratio * 1.5) * (1 if inventory_deviation > 0 else -1)
        
        # 计算新订单价格 (带有库存偏移的中间价)
        mid_price = md.midPrice
        half_spread = base_spread * spread_multiplier / 2
        
        # 计算买卖价格时考虑库存偏移
        bid_price = round(mid_price * (1 - half_spread - skew_factor * half_spread), 5)
        ask_price = round(mid_price * (1 + half_spread - skew_factor * half_spread), 5)
        
        # 根据价格动态计算订单大小
        base_bid_size = self.calculate_order_size(symbol, bid_price)
        base_ask_size = self.calculate_order_size(symbol, ask_price, is_ask=True)
        
        # 根据当前持仓调整订单大小
        if inventory_deviation > 0:  # 多头持仓过多
            # 增加卖单大小，减少买单大小
            ask_size = base_ask_size * (1 + inventory_ratio)
            bid_size = base_bid_size * (1 - inventory_ratio * 0.8)
        elif inventory_deviation < 0:  # 空头持仓过多
            # 增加买单大小，减少卖单大小
            bid_size = base_bid_size * (1 + inventory_ratio)
            ask_size = base_ask_size * (1 - inventory_ratio * 0.8)
        else:
            bid_size = base_bid_size
            ask_size = base_ask_size
        
        # 检查是否需要主动调整库存（定期或库存严重偏离时）
        should_rebalance = (
            md.timestamp > self.mm_last_rebalance.get(symbol, datetime(2000,1,1)) + self.mm_rebalance_interval
            or inventory_ratio > self.mm_inventory_limit
        )
        
        if should_rebalance and inventory_deviation != 0:
            self.evt.consoleLog(f"执行库存调整: {symbol} 当前={self.current_position[symbol]}, 目标={self.mm_inventory_target}")
            
            # 计算需要调整的数量
            rebalance_volume = abs(inventory_deviation) * 0.3  # 每次调整30%的偏差
            rebalance_volume = max(0.01, min(rebalance_volume, abs(inventory_deviation)))
            rebalance_volume = round(rebalance_volume, 2)
            
            # 确定调整方向
            rebalance_side = -1 if inventory_deviation > 0 else 1
            
            # 通过市价单主动调整库存
            if rebalance_volume >= 0.01:
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=f"{symbol}_REBALANCE",
                    volume=rebalance_volume,
                    openclose='close' if rebalance_side != 1 else 'open',
                    buysell=rebalance_side,
                    ordertype=0  # 市价单
                )
                self.evt.sendOrder(order)
                self.evt.consoleLog(f"主动库存调整: {symbol} {'买入' if rebalance_side > 0 else '卖出'} {rebalance_volume:.2f}手")
            
            # 更新最后调整时间
            self.mm_last_rebalance[symbol] = md.timestamp
            
        # 最终的订单大小限制在合理范围内
        bid_size = max(0.01, min(bid_size, self.max_position))
        ask_size = max(0.01, min(ask_size, self.max_position))
        
        # 挂双向限价单
        orders = [
            (1, bid_price, bid_ref, round(bid_size, 2)),   # 买单
            (-1, ask_price, ask_ref, round(ask_size, 2))   # 卖单
        ]
        
        for buysell, price, ref, size in orders:
            # 如果持仓超过极限，不再增加该方向的仓位
            if (buysell == 1 and self.current_position[symbol] >= self.max_position) or \
               (buysell == -1 and self.current_position[symbol] <= -self.max_position):
                continue
                
            # 如果订单量太小，不发送
            if size < 0.01:
                continue
                
            order = AlgoAPIUtil.OrderObject(
                instrument=symbol,
                orderRef=ref,
                volume=size,
                openclose='open',
                buysell=buysell,
                ordertype=1,          # Limit order
                timeinforce=self.mm_order_lifespan,  # 动态设置有效期
                price=price
            )
            self.evt.sendOrder(order)
            
            # 记录订单状态
            self.order_status[ref] = {
                'symbol': symbol,
                'buysell': buysell,
                'price': price,
                'size': size,
                'timestamp': md.timestamp
            }
            
        # 记录当前挂单状态
        self.evt.consoleLog(f"做市挂单: {symbol} 买入={bid_price:.5f}×{bid_size:.2f}, 卖出={ask_price:.5f}×{ask_size:.2f}, " 
                          f"价差={base_spread*spread_multiplier:.6f}, 偏移={skew_factor:.3f}")

    def close_all_market_making_positions(self):
        """平仓所有做市策略的持仓"""
        # 取消所有未成交的做市限价单
        for symbol in self.symbols:
            # 生成做市订单的引用格式
            for suffix in ["_BID", "_ASK"]:
                order_ref = f"{symbol}{suffix}"
                # 发送订单关闭请求
                cancel_order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    orderRef=order_ref,
                    openclose='close'  # 关闭指定订单
                )
                self.evt.sendOrder(cancel_order)
                self.evt.consoleLog(f"取消做市订单: {order_ref}")

        # 平仓现有做市持仓（市价单对冲）
        for symbol in self.symbols:
            current_pos = self.current_position.get(symbol, 0)
            if current_pos != 0:
                # 确定对冲方向
                close_side = -1 if current_pos > 0 else 1
                close_volume = abs(current_pos)
                # 发送市价单平仓
                order = AlgoAPIUtil.OrderObject(
                    instrument=symbol,
                    openclose='close',
                    buysell=close_side,
                    ordertype=0,  # 市价单
                    volume=close_volume,
                    orderRef=f"MM_CLOSE_{symbol}"
                )
                self.evt.sendOrder(order)
                self.evt.consoleLog(f"市价平仓做市持仓: {symbol} {close_side>0 and '买入' or '卖出'} {close_volume}手")
                # 更新本地持仓记录
                self.current_position[symbol] = 0
                self.position_cost[symbol] = 0

    # 关闭之前策略的所有仓位
    def close_all_pre_positions(self, pre_market_regime):
        if pre_market_regime == "trending":
            self.close_all_ma_cross_positions()
        else:
            self.close_all_market_making_positions()

    # 核心策略决策
    def check_trading_strategy(self, symbol, md):
        """检查应该应用哪种交易策略"""
        # 每5分钟更新一次策略
        if md.timestamp < self.last_update[symbol] + timedelta(minutes=5):
            return
        
        self.pre_market_regime = self.market_regime
        self.market_regime = self.determine_market_regime(md.timestamp)
        
        if self.pre_market_regime != self.market_regime:
            self.close_all_pre_positions(self.pre_market_regime)

        # 更新资金分配
        self.distribute_capital()
        
        # 更新与历史收益率相关的风险指标
        if len(self.price_history[symbol]) > 1:
            prices = list(self.price_history[symbol])
            recent_return = (prices[-1] / prices[-2]) - 1
            self.hist_returns[symbol].append(recent_return)

        # 基于市场状态选择策略
        # 趋势市场使用MA Cross，其他情况用做市策略（根据市场状态调整）
        if self.market_regime == "trending":
            self.ma_cross_active = True
            self.check_ma_cross_signal(symbol, md.timestamp)
        else:
            self.ma_cross_active = False
            self.place_market_making_orders(symbol, md)

        self.last_update[symbol] = md.timestamp

    # VaR风险控制
    def calculate_portfolio_var(self, confidence=0.95):
        """计算投资组合在险价值（历史模拟法）"""
        portfolio_returns = []
        
        # 生成组合收益率序列
        for symbol in self.symbols:
            pos = self.current_position.get(symbol, 0)
            if pos == 0 or symbol not in self.hist_returns:
                continue
                
            # 获取品种历史收益率
            rets = np.array(self.hist_returns[symbol])
            if len(rets) < 20:  # 数据不足时跳过
                continue
                
            # 计算持仓贡献的收益率
            pos_value = pos * self.price_table[symbol] 
            weight = pos_value / self.availableBalance if self.availableBalance>0 else 0
            portfolio_returns.append(weight * rets)
        
        if not portfolio_returns:
            return 0.0
            
        # 合并收益率（假设各品种收益率独立）
        combined_rets = np.sum(portfolio_returns, axis=0)
        
        # 计算VaR（历史分位数）
        var = np.percentile(combined_rets, (1-confidence)*100)
        return abs(var * self.availableBalance)  # 返回绝对风险值

    def reduce_risk_exposure(self, ratio=0.3):
        """按比例降低风险暴露"""
        self.evt.consoleLog(f"触发VaR风控: 开始减仓{ratio*100}%")
        
        for symbol in self.symbols:
            current_pos = self.current_position.get(symbol, 0)
            if current_pos == 0:
                continue
                
            # 计算减仓量
            reduce_vol = abs(current_pos) * ratio
            reduce_vol = round(reduce_vol, 2)
            
            # 发送对冲订单
            hedge_side = -1 if current_pos > 0 else 1
            order = AlgoAPIUtil.OrderObject(
                instrument=symbol,
                openclose='close',
                buysell=hedge_side,
                volume=reduce_vol,
                ordertype=0  # 市价单
            )
            self.evt.sendOrder(order)
            
            # 更新本地记录
            self.current_position[symbol] -= hedge_side * reduce_vol

    def on_bulkdatafeed(self, isSync, bd, ab):
        """处理批量数据更新"""
        if not isSync:
            return
            
        # 更新所有品种的价格
        for symbol in bd:
            if symbol not in self.symbols:
                self.initialize_symbol(symbol)
            mid_price = (bd[symbol]['bidPrice'] + bd[symbol]['askPrice']) / 2
            self.price_history[symbol].append(mid_price)
            # 更新价格表
            self.price_table[symbol] = mid_price

    def on_marketdatafeed(self, md, ab):
        """处理市场数据更新"""
        # 确保品种已初始化
        symbol = md.instrument
        if symbol not in self.symbols:
            self.initialize_symbol(symbol)

        # 更新价格表
        self.price_history[symbol].append(md.midPrice)
        self.price_table[symbol] = md.midPrice

        # 执行策略
        self.check_trading_strategy(symbol, md)

    def on_newsdatafeed(self, nd):
        pass

    def on_weatherdatafeed(self, wd):
        pass
    
    def on_econsdatafeed(self, ed):
        pass
        
    def on_corpAnnouncement(self, ca):
        pass

    def on_orderfeed(self, of):
        """订单反馈处理，更新持仓"""
        symbol = of.instrument
        if symbol in self.symbols and of.status == 'Filled':
            # update current capital
            res = self.evt.getAccountBalance()
            self.availableBalance = res["availableBalance"]
            self.evt.consoleLog("current availableBalance=",self.availableBalance)

            if of.orderRef in self.order_status:  # market making
                order_info = self.order_status[of.orderRef]
                
                # 确保更新的是订单对应的品种
                if symbol != order_info['symbol']:
                    self.evt.consoleLog(f"警告: 订单品种不匹配! 订单: {symbol}, 记录: {order_info['symbol']}")
                    return
                
                # 更新持仓
                position_change = of.filledQty * order_info['buysell']
                old_position = self.current_position[symbol]
                self.current_position[symbol] += position_change
                
                # 更新持仓成本 (加权平均)
                if position_change != 0:
                    if old_position == 0:
                        self.position_cost[symbol] = of.avgPrice
                    elif (old_position > 0 and position_change > 0) or (old_position < 0 and position_change < 0):
                        # 增加现有仓位
                        self.position_cost[symbol] = (self.position_cost[symbol] * abs(old_position) + of.avgPrice * abs(position_change)) / abs(self.current_position[symbol])
                    elif abs(position_change) > abs(old_position):
                        # 翻转仓位
                        self.position_cost[symbol] = of.avgPrice
                    else:
                        # 减少仓位但未翻转
                        # 成本保持不变
                        pass
                
                # 记录成交日志
                direction = "买入" if order_info['buysell'] > 0 else "卖出"
                self.evt.consoleLog(f"订单成交: 品种: {symbol}, {direction} {of.filledQty} @ {of.avgPrice:.5f}, " 
                                  f"当前持仓: {self.current_position[symbol]}, 持仓成本: {self.position_cost[symbol]:.5f}")
            
            # MA Cross订单成交
            elif of.orderRef and of.orderRef.startswith(('MA_', 'CLOSE_')):
                self.evt.consoleLog(f"策略订单成交: {of.orderRef}, 品种: {symbol}, 数量: {of.filledQty}, 价格: {of.avgPrice:.5f}")
                
                # 更新持仓
                if of.buysell:  # 确保buysell存在
                    position_change = of.filledQty * of.buysell
                    old_position = self.current_position.get(symbol, 0)
                    self.current_position[symbol] = old_position + position_change
                    
                    # 更新持仓成本
                    if old_position == 0:
                        self.position_cost[symbol] = of.avgPrice
                    elif (old_position > 0 and position_change > 0) or (old_position < 0 and position_change < 0):
                        self.position_cost[symbol] = (self.position_cost[symbol] * abs(old_position) + of.avgPrice * abs(position_change)) / abs(self.current_position[symbol])
                    elif abs(position_change) > abs(old_position):
                        self.position_cost[symbol] = of.avgPrice

    def on_dailyPLfeed(self, pl):
        """处理每日盈亏信息"""
        # 更新账户余额
        res = self.evt.getAccountBalance()
        self.availableBalance = res["availableBalance"]
        self.evt.consoleLog(f"日盈亏更新后余额: {self.availableBalance:.2f}")
        # 显示每个交易品种的盈亏状况
        total_unrealized_pnl = 0
        
        for symbol in self.symbols:
            if len(self.price_history[symbol]) > 0:
                last_price = self.price_history[symbol][-1]
                unrealized_pnl = self.current_position[symbol] * (last_price - self.position_cost[symbol])
                total_unrealized_pnl += unrealized_pnl
                
                self.evt.consoleLog(f"品种: {symbol}, 持仓量: {self.current_position[symbol]}, " 
                                   f"持仓成本: {self.position_cost[symbol]:.5f}, 未实现盈亏: {unrealized_pnl:.2f}")
        
        # 更新资金分配情况
        self.distribute_capital()
        
        # 显示市场状态
        self.evt.consoleLog(f"当前市场状态: {self.market_regime}, MA Cross状态: {'激活' if self.ma_cross_active else '未激活'}")
        
        # 显示当前账户资金和资金分配情况
        pct_allocated = (self.total_allocated_capital / self.availableBalance * 100) if self.availableBalance > 0 else 0
        self.evt.consoleLog(f"当前账户资金: {self.availableBalance:.2f}, 已分配资金: {self.total_allocated_capital:.2f} ({pct_allocated:.1f}%)")
        
        # 修复: pl是字典类型
        if isinstance(pl, dict) and 'tradingDay' in pl and 'dailyPL' in pl:
            self.evt.consoleLog(f"日期: {pl['tradingDay']}, 已实现盈亏: {pl['dailyPL']:.2f}, " 
                               f"总未实现盈亏: {total_unrealized_pnl:.2f}")
        else:
            self.evt.consoleLog(f"日盈亏更新, 总未实现盈亏: {total_unrealized_pnl:.2f}")

        if pl.get('Acdate', datetime.now()) > self.last_var_check + self.var_check_interval:
            current_var = self.calculate_portfolio_var()
            self.evt.consoleLog(f"VaR监控: 当前风险值={current_var:.2f}, 净值占比={(current_var/self.availableBalance):.2%}")
            
            # 触发风险控制
            if current_var > self.availableBalance * self.var_threshold:
                self.reduce_risk_exposure()
                
            self.last_var_check = pl.get('Acdate', datetime.now())

    def on_openPositionfeed(self, op, oo, uo):
        """处理开仓位置信息"""
        self.evt.consoleLog(f"收到持仓更新: {op}")

        # 更新账户余额
        res = self.evt.getAccountBalance()
        self.availableBalance = res["availableBalance"]
        self.evt.consoleLog(f"持仓更新后余额: {self.availableBalance:.2f}")
        
        # 尝试解析持仓信息
        try:
            # 假设op是字典列表，每个字典包含持仓信息
            for position in op:
                if isinstance(position, dict) and 'instrument' in position:
                    symbol = position['instrument']
                    
                    # 确保品种已初始化
                    if symbol not in self.symbols:
                        self.initialize_symbol(symbol)
                    
                    self.current_position[symbol] = position.get('volume', 0)
                    self.position_cost[symbol] = position.get('averagePrice', 0)
                    self.evt.consoleLog(f"持仓更新: {symbol}, 数量: {self.current_position[symbol]}, " 
                                       f"成本: {self.position_cost[symbol]:.5f}")
        except Exception as e:
            # 如果无法按预期解析，记录错误
            self.evt.consoleLog(f"解析持仓信息错误: {str(e)}")
            self.evt.consoleLog(f"原始持仓数据: {op}")
