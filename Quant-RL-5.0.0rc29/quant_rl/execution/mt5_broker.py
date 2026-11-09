"""One MT5 terminal interface for every LIVE account. No demo/funded branches."""
from __future__ import annotations
from datetime import datetime, timezone
import math
import time
from quant_rl.data.mt5_source import MT5DataSource, account_is_demo


class MT5Broker(MT5DataSource):
    def __init__(self, connection, *, allow_non_demo_account: bool = False):
        super().__init__(login=connection.mt5_login,
            password=connection.mt5_password, server=connection.mt5_server,
            path=connection.mt5_path)
        self.allow_non_demo_account = bool(allow_non_demo_account)

    def connect(self):
        super().connect()
        account = self.account()
        if self.login and int(account.login) != self.login:
            self.close()
            raise ConnectionError('Connected MT5 login does not match requested login')

    def account(self):
        if self._mt5 is None:
            raise ConnectionError('MT5 terminal not connected')
        terminal=self._mt5.terminal_info()
        account=self._mt5.account_info()
        if terminal is None or account is None or not bool(terminal.connected):
            raise ConnectionError('MT5 terminal disconnected')
        if not bool(account.trade_allowed) or not bool(terminal.trade_allowed):
            raise PermissionError('MT5 trading is not permitted by terminal/account')
        if getattr(account, 'trade_mode', None) is None:
            raise PermissionError('MT5 account trading mode unavailable; LIVE is blocked')
        if not self.allow_non_demo_account and not account_is_demo(account, self._mt5):
            raise PermissionError('LIVE is Demo-only; explicitly enable live.allow_non_demo_account to use a funded account')
        if not math.isfinite(float(account.equity)) or float(account.equity)<=0:
            raise ValueError('Invalid broker-reported account equity')
        return account

    def tick(self,symbol):
        if not self._mt5.symbol_select(symbol,True):
            raise ConnectionError('MT5 symbol selection failed')
        tick=self._mt5.symbol_info_tick(symbol)
        if tick is None or not all(math.isfinite(float(getattr(tick,k,0))) and float(getattr(tick,k,0))>0 for k in ('bid','ask')):
            raise ConnectionError('No valid live bid/ask quotes')
        if tick.ask<tick.bid:
            raise ConnectionError('Invalid bid/ask ordering')
        return tick

    def positions(self):
        result=self._mt5.positions_get()
        if result is None:
            raise ConnectionError('Could not query all MT5 positions')
        return result

    def orders(self):
        result=self._mt5.orders_get()
        if result is None:
            raise ConnectionError('Could not query MT5 pending orders')
        return result

    def recent_deals(self, start_s):
        # MT5 history API accepts UTC datetimes and returns UTC epoch timestamps.
        result=self._mt5.history_deals_get(
            datetime.fromtimestamp(start_s,timezone.utc),
            datetime.fromtimestamp(time.time(),timezone.utc))
        if result is None:
            raise ConnectionError('Could not query MT5 trade history')
        return result

    def order_request(self, *, symbol, direction, volume, price, sl, tp, deviation, magic, comment):
        m=self._mt5
        info=m.symbol_info(symbol)
        if info is None:
            raise ConnectionError('Missing broker symbol details')
        # SYMBOL_FILLING_MODE is a bitmask; 1=FOK, 2=IOC. RETURN is
        # prohibited under MARKET execution. Never guess unsupported modes.
        flags=int(info.filling_mode)
        if flags & 2:
            filling=m.ORDER_FILLING_IOC
        elif flags & 1:
            filling=m.ORDER_FILLING_FOK
        elif int(info.trade_exemode)!=int(getattr(m,'SYMBOL_TRADE_EXECUTION_MARKET',2)):
            filling=m.ORDER_FILLING_RETURN
        else:
            raise ValueError('MT5 symbol does not advertise a supported filling mode')
        return dict(action=m.TRADE_ACTION_DEAL,symbol=symbol,
            type=(m.ORDER_TYPE_BUY if direction=='BUY' else m.ORDER_TYPE_SELL),
            volume=float(volume),price=float(price),sl=float(sl),tp=float(tp),
            deviation=int(deviation),magic=int(magic),comment=comment,
            type_time=m.ORDER_TIME_GTC,type_filling=filling)

    def submit(self, request):
        checked=self._mt5.order_check(request)
        if checked is None or int(getattr(checked,'retcode',-1)) != 0:
            code=None if checked is None else getattr(checked,'retcode',None)
            return {'status':'CHECK_REJECTED','retcode':code}
        result=self._mt5.order_send(request)
        if result is None:
            # Unknown state: broker may have accepted before connection dropped.
            raise ConnectionError('MT5 order outcome unknown; do not retry automatically')
        code=int(result.retcode)
        if code!=int(self._mt5.TRADE_RETCODE_DONE):
            return {'status':'REJECTED_OR_UNCERTAIN','retcode':code}
        return {'status':'FILLED','retcode':code,'order':int(getattr(result,'order',0)),
                'deal':int(getattr(result,'deal',0))}

    def closed_bars(self,symbol, count=2400):
        # H1 history for the shared research observation builder. The existing
        # source removes incomplete current candles, including around session gaps.
        return self.fetch_bars(symbol,'H1',count)
