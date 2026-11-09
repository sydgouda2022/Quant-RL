"""Reconciled per-symbol financial reports, including exact veto and trade ledgers.

PDF charts are drawn as ReportLab vector graphics to avoid loading a second
OpenMP-enabled plotting stack alongside PyTorch. Learning penalties are not cash.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import numpy as np
from quant_rl.execution.accounting import ExecutedTradeRecord, summarize_veto_context


def utc(seconds: int) -> str:
    return datetime.fromtimestamp(int(seconds), timezone.utc).isoformat()


def _csv(path: Path, rows, cols):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def _monthly_curve(sim):
    monthly={}
    for stamp,equity in zip(sim.time_s_curve,sim.equity_curve):
        monthly[datetime.fromtimestamp(int(stamp),timezone.utc).strftime('%Y-%m')]=float(equity)
    previous=float(sim.starting_balance)
    rows=[]
    for month,value in sorted(monthly.items()):
        rows.append({'month_utc':month,'ending_equity':value,
                     'net_return_pct':100*(value/previous-1) if previous>0 else 0})
        previous=value
    return rows


def _streaks(trades):
    wins=losses=curw=curl=0
    for t in trades:
        if t.net_pnl_account_ccy>0:
            curw+=1;curl=0
        else:
            curl+=1;curw=0
        wins=max(wins,curw);losses=max(losses,curl)
    return wins,losses


def write_economic_report(root,symbol,sim,*,origin,settings,period,qualified,reasons,
                          trial_report,provenance=None):
    import uuid
    output=(Path(root)/'reports'/symbol/
            (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]))
    output.mkdir(parents=True,exist_ok=False)
    trades=[asdict(t) for t in sim.executed_trades]
    decisions=[]; all_failed=Counter(); proposed=Counter()
    for event in sim.decision_events:
        d=asdict(event)
        d['measured']=dict(event.measured)
        d['observed_features']=dict(event.observed_features)
        d['failed_check_details']=dict(event.failed_check_details)
        decisions.append(d)
        proposed[event.action]+=1
        for code in event.veto_codes:all_failed[code]+=1
    _csv(output/'trades.csv',trades,[f.name for f in fields(ExecutedTradeRecord)])
    _csv(output/'decisions.csv',[
        {**d,'veto_codes':';'.join(d['veto_codes']),
         'failed_check_details':json.dumps(d['failed_check_details'],sort_keys=True),
         'measured':json.dumps(d['measured'],sort_keys=True),
         'observed_features':json.dumps(d['observed_features'],sort_keys=True)} for d in decisions
    ],list(decisions[0]) if decisions else ['bar_index','decision_time_s','action','status'])
    per_check = summarize_veto_context(sim.decision_events)
    check_fields = ['check','failure_count','mean_observed_spread_pips',
                    'mean_observed_volatility_price','mean_proposed_stop_units',
                    'mean_proposed_target_units','mean_proposed_risk_fraction',
                    'mean_round_trip_cost_pips','mean_allowed_cost_pips',
                    'mean_raw_requested_lots','mean_broker_min_volume_lots',
                    'mean_broker_volume_step_lots','mean_requested_stop_pips',
                    'mean_max_feasible_stop_pips_at_requested_risk',
                    'mean_required_risk_fraction_for_min_lot_at_requested_stop',
                    'mean_volume_shortfall_lots']
    _csv(output/'veto_by_check.csv', [
        {'check':code, **stats} for code,stats in per_check.items()], check_fields)
    _csv(output/'optimization_trials.csv', [{
        'trial': r.get('trial'),
        'validation_learning_reward':r.get('validation_reward_account_ccy'),
        'feature_group':r.get('parameters',{}).get('feature_group'),
        'sequence_bars':r.get('parameters',{}).get('sequence_bars'),
        'vetoed_proposals':r.get('decision_diagnostics',{}).get('vetoed_proposals'),
        'no_trade_decisions':r.get('decision_diagnostics',{}).get('no_trade_decisions'),
        'executed_proposals':r.get('decision_diagnostics',{}).get('executed_proposals'),
        'failed_check_counts':json.dumps(r.get('decision_diagnostics',{}).get('all_failed_check_counts',{}),sort_keys=True),
        'veto_context_by_check':json.dumps(r.get('decision_diagnostics',{}).get('veto_context_by_check',{}),sort_keys=True),
    } for r in trial_report],
    ['trial','validation_learning_reward','feature_group','sequence_bars',
     'vetoed_proposals','no_trade_decisions','executed_proposals','failed_check_counts','veto_context_by_check'])
    _csv(output/'equity.csv',[
        {'time_utc':utc(t),'equity':float(value)}
        for t,value in zip(sim.time_s_curve,sim.equity_curve)],['time_utc','equity'])
    months=_monthly_curve(sim)
    _csv(output/'monthly_returns.csv',months,['month_utc','ending_equity','net_return_pct'])
    winners=[t for t in sim.executed_trades if t.net_pnl_account_ccy>0]
    losers=[t for t in sim.executed_trades if t.net_pnl_account_ccy<=0]
    gross=sum(t.gross_pnl_account_ccy for t in sim.executed_trades)
    costs={k:sum(getattr(t,k) for t in sim.executed_trades) for k in
           ('spread_cost_account_ccy','commission_cost_account_ccy',
            'slippage_cost_account_ccy','financing_cost_account_ccy')}
    realized=sum(t.net_pnl_account_ccy for t in sim.executed_trades)
    if abs(sim.starting_balance+realized-sim.ending_balance)>0.01:
        raise AssertionError('Ending realized balance fails trade ledger reconciliation')
    if abs(gross-sum(costs.values())-realized)>0.01:
        raise AssertionError('Gross P&L, transaction costs, and net P&L fail reconciliation')
    # Learning reward is based on *equity*, including any censored position's
    # boundary mark, whereas brokerage cash balance reflects closed trades.
    if abs((sim.ending_equity-sim.starting_balance)
           -sim.penalty_total_account_ccy-sim.learning_reward_account_ccy)>0.01:
        raise AssertionError('Learning reward fails marked-to-market reconciliation')
    available_rewards=sum(e.reward_account_ccy for e in sim.decision_events)
    if abs(available_rewards-sim.learning_reward_account_ccy)>0.01:
        raise AssertionError('Decision rewards do not reconcile with total learning reward')
    if abs(sim.ending_balance+sum(p.unrealized_pnl_account_ccy for p in sim.open_positions_at_end)
           -sim.ending_equity)>0.01:
        raise AssertionError('Final equity does not reconcile with open positions')
    n=len(trades); win_streak,loss_streak=_streaks(sim.executed_trades)
    per_direction={}
    for direction in ('BUY','SELL'):
        subset=[t for t in sim.executed_trades if str(t.direction)==direction]
        per_direction[direction]={'trades':len(subset),'net_pnl':sum(t.net_pnl_account_ccy for t in subset),
            'win_rate':sum(t.net_pnl_account_ccy>0 for t in subset)/len(subset) if subset else 0.}
    summary={
        'symbol':symbol,'origin':origin,
        'provenance':provenance or {},'period':period,'qualified':qualified,
        'qualification_reasons':reasons,'settings':settings,
        'starting_balance':sim.starting_balance,'ending_balance':sim.ending_balance,
        'ending_equity':sim.ending_equity,'unrealized_pnl':sim.ending_equity-sim.ending_balance,
        'gross_realized_pnl':gross,'net_realized_pnl':realized,
        'cost_breakdown':costs,'total_trading_costs':sum(costs.values()),
        'financing_sources':sorted(set(t.financing_source for t in sim.executed_trades)) or ['MODELED_CONFIG'],
        'financing_rollover_units_total':sum(t.financing_rollover_units for t in sim.executed_trades),
        'net_return_pct':sim.net_return_pct,'cagr_pct':sim.compounded_annual_growth_pct,
        'max_drawdown_pct':sim.max_drawdown_pct,'calmar_ratio':sim.calmar_ratio,
        'sharpe_ratio':sim.sharpe_ratio,'sortino_ratio':sim.sortino_ratio,
        'closed_trades':n,'winning_trades':len(winners),'losing_trades':len(losers),
        'win_rate':sim.win_rate,'profit_factor':sim.profit_factor,'mean_net_r':sim.mean_net_r,
        'net_expectancy_per_trade_account_ccy':realized/n if n else 0,
        'average_win_account_ccy':sum(t.net_pnl_account_ccy for t in winners)/len(winners) if winners else 0,
        'average_loss_account_ccy':sum(t.net_pnl_account_ccy for t in losers)/len(losers) if losers else 0,
        'average_holding_days':sum(t.holding_days for t in sim.executed_trades)/n if n else 0,
        'average_risk_fraction':sum(t.requested_risk_fraction for t in sim.executed_trades)/n if n else 0,
        'maximum_winning_streak':win_streak,'maximum_losing_streak':loss_streak,
        'trades_per_year':sim.trades_per_year,'capital_utilization_pct':sim.capital_utilization_pct,
        'abstention_rate':sim.abstention_rate,'decision_counts':dict(proposed),
        'veto_count':sim.veto_count,'primary_veto_breakdown':sim.veto_breakdown,
        'all_failed_check_counts':dict(all_failed),'veto_context_by_check':per_check,
        'by_direction':per_direction,
        'inactivity_and_veto_learning_penalties':sim.penalty_total_account_ccy,
        'total_learning_reward':sim.learning_reward_account_ccy,
        'terminal_censored_learning_mark':sum(
            e.terminal_mark_to_market_account_ccy or 0. for e in sim.decision_events),
        'unresolved_positions':[asdict(p) for p in sim.open_positions_at_end],
        'terminal_halt_reason':sim.terminal_halt_reason,
        'terminal_halt_bar_index':sim.terminal_halt_bar_index,
        'monthly_performance':months,'trial_report':trial_report,
        'disclaimer':'Model evaluation is simulated; no verified live fills or future profit claims.'}
    (output/'summary.json').write_text(
        json.dumps(summary,indent=2,sort_keys=True,default=str,allow_nan=False),encoding='utf-8')
    metric_keys=['starting_balance','ending_balance','ending_equity','unrealized_pnl',
        'gross_realized_pnl','total_trading_costs','net_realized_pnl','net_return_pct',
        'cagr_pct','max_drawdown_pct','sharpe_ratio','sortino_ratio','win_rate',
        'profit_factor','mean_net_r','net_expectancy_per_trade_account_ccy','closed_trades',
        'trades_per_year','capital_utilization_pct','veto_count',
        'inactivity_and_veto_learning_penalties','total_learning_reward',
        'terminal_censored_learning_mark']
    md=[f'# {symbol} | Economic Performance Report', '',
        '**MT5 HISTORICAL REPLAY - NOT LIVE TRADING**', '',
        f"Period: {period['test_start_utc']} to {period['test_end_utc']}",
        f"Qualified: {'yes' if qualified else 'no'}; Reasons: {', '.join(reasons) or 'none'}",'',
        '## Economic scorecard','', '| Measurement | Value |','|---|---:|']
    md += [f'| {k.replace("_"," ").title()} | {summary[k]:,.4f} |' for k in metric_keys]
    md += ['', '## Transaction costs', '', '| Expense | Amount |','|---|---:|']
    md += [f'| {k.replace("_"," ").title()} | {v:,.4f} |' for k,v in costs.items()]
    md += ['', '## Financing evidence', '',
       '- Model sources: ' + ', '.join(summary['financing_sources']),
       f"- Total weighted rollover units across closed trades: {summary['financing_rollover_units_total']}",
       '- UTC rollover assumptions and account-currency cash inputs are unverified until reconciled against actual broker deals.',
       '', '## Monthly returns', '', '| Month UTC | Net return | Ending equity |','|---|---:|---:|']
    md += [f"| {m['month_utc']} | {m['net_return_pct']:+.2f}% | ${m['ending_equity']:,.2f} |" for m in months]
    md += ['', '## Examiner failures (all violated checks)', '', '| Check | Failures |','|---|---:|']
    md += [f'| {k} | {v} |' for k,v in sorted(all_failed.items())]
    md += ['', '## Trial comparison (validation economics and Examiner)',
           '', '| Trial | Validation learning reward ($) | Vetoes | Most frequent failed check | Feature group |',
           '|---|---:|---:|---|---|']
    for r in trial_report:
        diagnosis=r.get('decision_diagnostics',{})
        counts=diagnosis.get('all_failed_check_counts',{})
        top=max(sorted(counts), key=lambda c:counts[c]) if counts else 'none'
        md.append(f"| {r.get('trial')} | {r.get('validation_reward_account_ccy','failed')} | "
                  f"{diagnosis.get('vetoed_proposals','n/a')} | {top} | "
                  f"{r.get('parameters',{}).get('feature_group','n/a')} |")
    md += ['', 'Trial-level failure statistics are correlations for further research, not proof that a specific indicator caused a veto.']
    md += ['', '## Historical spread assumptions', '',
       f"- Broker-reported H1 spread coverage: {summary['provenance'].get('reported_spread_fraction',0.0):.2%}",
       f"- Missing-spread estimate: {summary['provenance'].get('fallback_spread_points')}",
       f"- Estimate source: {summary['provenance'].get('fallback_spread_source', 'NONE')}",
       '- Estimated costs are not historical broker-observed spreads; actual fills remain unverified.',
       '', '## M1 intrabar evidence', '',
       f"- Minute spread coverage: {summary['provenance'].get('m1_spread_coverage_fraction', 0.0):.2%}",
       f"- Fully time-covered H1 input bars: {summary['provenance'].get('m1_intrabar_coverage_fraction', 0.0):.2%}",
       '- These statistics do not certify tick-level spread or actual execution prices.',
       '', '## Data and audit trail', '',
       '- `trades.csv`: every closed trade with P&L, stop, target, size, and expenses',
       '- `decisions.csv`: every eligible proposal, all failed check explanations, thresholds and penalties',
       '- `optimization_trials.csv`: hyperparameter trials with validation veto breakdowns'
       ,'- `veto_by_check.csv`: aggregated final evaluation rejection context for every failed C1-C26 check',
       '- `equity.csv`: chronological marked-to-market equity',
       '- `monthly_returns.csv`: monthly equity returns',
       '- `summary.json`: full machine-readable report and provenance',
       '- `report.pdf`: printable comprehensive economic report', '',
       '**Learning penalties are not broker charges. A censored position contributes its observable terminal mark to learning, but is never forcibly closed.**']
    (output/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    _render_pdf(output/'report.pdf',summary,sim)
    return output


def _render_pdf(path: Path, x: dict, sim):
    """A clean multi-page PDF, derived entirely from reconciled report values."""
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Line, PolyLine, Polygon, String
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak)

    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='QRTitle',parent=styles['Title'],fontSize=20,leading=24,
                              textColor=colors.HexColor('#173e60'),spaceAfter=9))
    styles.add(ParagraphStyle(name='QRSection',parent=styles['Heading2'],fontSize=12,
                              textColor=colors.HexColor('#23587b'),spaceBefore=12,spaceAfter=6))
    styles.add(ParagraphStyle(name='QRBody',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=6))
    styles.add(ParagraphStyle(name='QRFoot',parent=styles['BodyText'],fontSize=8,
                              textColor=colors.HexColor('#596575'),leading=10))
    doc=SimpleDocTemplate(str(path),pagesize=(612,792),leftMargin=42,rightMargin=42,
                          topMargin=42,bottomMargin=40)
    story=[]
    label='MT5 HISTORICAL REPLAY / NOT LIVE'
    story += [Paragraph(f"QUANT-RL | {x['symbol']}",styles['QRTitle']),
              Paragraph('Economic performance and Examiner report',styles['Heading2']),
              Paragraph(label,styles['QRFoot']),Spacer(1,10)]
    def money(z):return f'${z:,.2f}'
    def percent(z):return f'{z:+,.2f}%'
    def add_table(rows,widths=None):
        t=Table([[str(y) for y in row] for row in rows],colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff5')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#173e60')),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.6),
            ('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#95b2c7')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
            ('VALIGN',(0,0),(-1,-1),'TOP')]))
        story.append(t)
    def section(s):story.append(Paragraph(s,styles['QRSection']))
    add_table([['Account','Starting','Ending balance','Ending equity'],
               [x['symbol'],money(x['starting_balance']),money(x['ending_balance']),
                money(x['ending_equity'])]],[108,136,140,144])
    section('Economic outcome')
    add_table([['Metric','Value','Metric','Value'],
      ['Net return',percent(x['net_return_pct']),'Max drawdown',f"{x['max_drawdown_pct']:.2f}%"],
      ['Net realized P&L',money(x['net_realized_pnl']),'Unrealized P&L',money(x['unrealized_pnl'])],
      ['Win rate',f"{x['win_rate']*100:.2f}%",'Profit factor',f"{x['profit_factor']:.2f}"],
      ['Closed trades',str(x['closed_trades']),'Expectancy',money(x['net_expectancy_per_trade_account_ccy'])],
      ['Total costs',money(x['total_trading_costs']),'Learning penalties',money(x['inactivity_and_veto_learning_penalties'])]
    ],[142,106,142,138])
    section('Marked-to-market equity and drawdown')
    # Draw the equity and drawdown curves directly with ReportLab. Importing
    # Matplotlib into a PyTorch-enabled Windows Conda process can load a second
    # Intel OpenMP DLL and crash the entire process with OMP Error #15.
    curves=np.asarray(sim.equity_curve,dtype=float)
    if not len(curves):
        curves=np.asarray([float(sim.starting_balance)])
    peaks=np.maximum.accumulate(np.maximum(curves,sim.starting_balance))
    dd=100*(peaks-curves)/np.maximum(peaks,1.)
    chart=Drawing(490,245)
    ink=colors.HexColor('#245778')
    grid=colors.HexColor('#d9e2ea')
    base=colors.HexColor('#9daebe')
    top_left=55.0
    top_width=424.0
    def _panel(y0,height,values,baseline=None,kind='equity'):
        bottom=y0+12
        usable=height-30
        arr=np.asarray(values,dtype=float)
        vmin=float(np.min(arr))
        vmax=float(np.max(arr))
        if baseline is not None:
            vmin=min(vmin,float(baseline))
            vmax=max(vmax,float(baseline))
        if vmax-vmin<1e-9:
            spread=max(1.,abs(vmax)*.002)
            vmin-=spread
            vmax+=spread
        pad=(vmax-vmin)*.08
        vmin=max(0.,vmin-pad) if kind=='drawdown' else vmin-pad
        vmax+=pad
        def yy(v):return bottom+usable*(float(v)-vmin)/(vmax-vmin)
        for fraction in (0.,.5,1.):
            y=bottom+usable*fraction
            chart.add(Line(top_left,y,top_left+top_width,y,strokeColor=grid,strokeWidth=.45))
            tick=vmin+(vmax-vmin)*fraction
            label=f'{tick:,.0f}' if kind=='equity' else f'{tick:.1f}%'
            chart.add(String(top_left-6,y-2.5,label,fontName='Helvetica',fontSize=7,
                             textAnchor='end',fillColor=ink))
        if baseline is not None:
            chart.add(Line(top_left,yy(baseline),top_left+top_width,yy(baseline),
                           strokeColor=base,strokeWidth=.6,strokeDashArray=[3,3]))
        # Limit vector size for multi-year backtests while keeping chronology.
        sample=np.unique(np.linspace(0,len(arr)-1,min(len(arr),700),dtype=int))
        coords=[(top_left+top_width*int(i)/max(1,len(arr)-1),yy(arr[int(i)])) for i in sample]
        if kind=='drawdown' and len(coords)>1:
            bottom_shape=[coords[0][0],bottom]
            for x,y in coords:bottom_shape.extend([x,y])
            bottom_shape.extend([coords[-1][0],bottom])
            chart.add(Polygon(bottom_shape,fillColor=colors.HexColor('#e0eaf2'),
                              strokeColor=None))
        if len(coords)>1:
            chart.add(PolyLine([v for pt in coords for v in pt],strokeColor=ink,strokeWidth=1.1))
        elif coords:
            chart.add(Line(coords[0][0],coords[0][1],coords[0][0]+.2,coords[0][1],
                           strokeColor=ink,strokeWidth=1.1))
    _panel(98,142,curves,sim.starting_balance,'equity')
    _panel(12,84,dd,0.,'drawdown')
    chart.add(String(top_left,236,'Equity (account currency)',fontName='Helvetica-Bold',
                     fontSize=8,fillColor=ink))
    chart.add(String(top_left,98,'Maximum-to-date drawdown',fontName='Helvetica-Bold',
                     fontSize=8,fillColor=ink))
    chart.add(String(top_left+top_width/2,3,'Chronological evaluation candles',
                     fontName='Helvetica',fontSize=7,fillColor=ink,textAnchor='middle'))
    story.append(chart)
    story.append(Paragraph('Final open positions are marked to market; they are not artificially closed.',styles['QRFoot']))
    story.append(PageBreak())

    story.append(Paragraph('Trading economics',styles['QRTitle']))
    section('Profit and loss reconciliation')
    cost=x['cost_breakdown']
    expense_rows=[['Component','Account-currency amount'],
                  ['Gross realized P&L',money(x['gross_realized_pnl'])]]
    expense_rows += [[k.replace('_account_ccy','').replace('_',' ').title(),
                      ('-'+money(v) if v >= 0 else '+'+money(-v))] for k,v in cost.items()]
    expense_rows += [['Net realized P&L',money(x['net_realized_pnl'])],
                     ['Unrealized open-position P&L',money(x['unrealized_pnl'])],
                     ['Ending equity',money(x['ending_equity'])]]
    add_table(expense_rows,[255,273])
    section('Trade characteristics')
    add_table([['Characteristic','Value','Characteristic','Value'],
        ['Wins / losses',f"{x['winning_trades']} / {x['losing_trades']}",'Mean net R',f"{x['mean_net_r']:+.3f}R"],
        ['Average winner',money(x['average_win_account_ccy']),'Average loser',money(x['average_loss_account_ccy'])],
        ['Longest winning streak',str(x['maximum_winning_streak']),'Longest losing streak',str(x['maximum_losing_streak'])],
        ['Average holding (days)',f"{x['average_holding_days']:.2f}",'Capital utilization',f"{x['capital_utilization_pct']:.2f}%"],
        ['CAGR',percent(x['cagr_pct']),'Trades / year',f"{x['trades_per_year']:.1f}"]
        ],[160,104,160,104])
    section('By direction')
    add_table([['Direction','Closed trades','Net P&L','Win rate']]+[
        [direction,str(info['trades']),money(info['net_pnl']),f"{info['win_rate']*100:.2f}%"]
        for direction,info in x['by_direction'].items()],[132,132,132,132])
    story.append(Paragraph(
        'Financing: ' + ', '.join(x['financing_sources']) +
        '; weighted UTC rollover units = ' + str(x['financing_rollover_units_total']) +
        '. Not broker-certified or tick-accurate.', styles['QRFoot']))
    story.append(Paragraph('Actual financial P&L excludes artificial inactivity/veto learning penalties.',styles['QRFoot']))
    story.append(PageBreak())

    story.append(Paragraph('Examiner and policy diagnostics',styles['QRTitle']))
    section('Decision outcomes')
    counts=x['decision_counts']
    add_table([['Action proposed','Count']]+[[k,str(v)] for k,v in sorted(counts.items())],[264,264])
    section('Rejections by failed safety check')
    failed=x['all_failed_check_counts']
    if failed:
        add_table([['Check','Failure count']]+[[k,str(v)] for k,v in sorted(failed.items())],[264,264])
    else:story.append(Paragraph('No Examiner vetoes occurred in this evaluation.',styles['QRBody']))
    if 'C16' in x.get('veto_context_by_check',{}):
        c16=x['veto_context_by_check']['C16']
        if 'mean_max_feasible_stop_pips_at_requested_risk' in c16:
            story.append(Paragraph(
                'C16 broker-volume diagnostics: mean affordable stop at requested risk '
                f"{c16['mean_max_feasible_stop_pips_at_requested_risk']:.1f} pips; "
                f"mean proposed stop {c16['mean_requested_stop_pips']:.1f} pips; "
                f"mean requested lots {c16['mean_raw_requested_lots']:.5f} versus "
                f"broker minimum {c16['mean_broker_min_volume_lots']:.5f}. "
                'These are research diagnostics, not approved replacement orders.',
                styles['QRFoot']))
    section('Reward reconciliation')
    add_table([['Quantity','Account-currency value'],
        ['Realized net trading P&L',money(x['net_realized_pnl'])],
        ['Censored open-position boundary P&L',money(x['terminal_censored_learning_mark'])],
        ['Penalty for eligible NO_TRADE and vetoes','-'+money(x['inactivity_and_veto_learning_penalties'])],
        ['Total training-style reward',money(x['total_learning_reward'])],
        ['Ending brokerage balance',money(x['ending_balance'])]],[296,232])
    if x.get('terminal_halt_reason'):
        story.append(Paragraph('Research account halted by '+x['terminal_halt_reason']+
            '. The model received no further decision opportunities or artificial penalties.',styles['QRFoot']))
    story.append(Paragraph('For each veto, decisions.csv records every failed C1-C26 check, its explanation, and the available measured quantities. All mandatory protections remain independent of model optimization.',styles['QRFoot']))
    story.append(PageBreak())

    story.append(Paragraph('Monthly performance and audit trail',styles['QRTitle']))
    section('Equity-based monthly returns')
    months=x['monthly_performance']
    if months:
        monthly_rows=[['Month (UTC)','Return','Closing equity']]+[
            [m['month_utc'],percent(m['net_return_pct']),money(m['ending_equity'])] for m in months]
        add_table(monthly_rows,[176,176,176])
    section('Evaluation provenance')
    story.append(Paragraph(f"Data source: {x['origin']}; final period: {x['period']['test_start_utc']} through {x['period']['test_end_utc']}",styles['QRBody']))
    story.append(Paragraph('Qualified: '+str(x['qualified'])+'; reasons: '+(', '.join(x['qualification_reasons']) or 'none'),styles['QRBody']))
    prov=x.get('provenance',{})
    story.append(Paragraph(
        f"H1 reported spread coverage: {100*prov.get('reported_spread_fraction',0.0):.1f}%; "
        f"missing-spread estimate: {prov.get('fallback_spread_points')} points; "
        f"basis: {prov.get('fallback_spread_source','NONE')}. "
        'Estimated costs are not actual historical fills.',styles['QRFoot']))
    story.append(Paragraph(
        f"M1 time coverage: {100*prov.get('m1_intrabar_coverage_fraction',0.0):.1f}%; "
        f"valid reported M1 spreads: {100*prov.get('m1_spread_coverage_fraction',0.0):.1f}%. "
        'Bar-reported spreads are not tick-level fill evidence.',styles['QRFoot']))
    if x.get('unresolved_positions'):
        story.append(Paragraph(f"Unresolved positions at boundary: {len(x['unresolved_positions'])}",styles['QRBody']))
    story.append(Paragraph('Source evidence: summary.json, trades.csv, decisions.csv, equity.csv, monthly_returns.csv. Research policy and economic simulator are reused for validation. Historical simulation does not prove live execution performance.',styles['QRFoot']))
    def on_page(canvas,document):
        canvas.setFont('Helvetica',7.5);canvas.setFillColor(colors.HexColor('#708090'))
        canvas.drawString(42,25,'QUANT-RL v5 | Simulated economic evaluation')
        canvas.drawRightString(568,25,f'Page {document.page}')
    doc.build(story,onFirstPage=on_page,onLaterPages=on_page)
