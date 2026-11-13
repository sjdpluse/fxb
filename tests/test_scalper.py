"""Software fixtures only: these tests establish no trading profitability."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from fxb.data import Bars, synthetic_bars, iso
from fxb.memory import Memory
from fxb.mt5 import MT5, MAGIC, Blocked
from fxb.execution import LiveAuthorization
from fxb.scalping import (analyze, FEATURES, VERSION, READOUT_DIM, ScalpConfig,
    scalp_risk, learning_state, entry_score, readout_context, learn_closed)
from fxb.scalper_runtime import create_profile, policy, scalp_cycle, settle_attempts, learning_scope
from test_mt5 import FakeMT5


def fixture(n=500):
    spec = replace(synthetic_bars(800).spec, symbol="XAUUSD.m", timeframe_seconds=60,
                   commission_per_lot_side=0, slippage_points=0)
    # The native API is mocked in execution tests. This is never market evidence.
    return Bars(1704067200 + np.arange(n)*60., np.full(n, 100.), np.full(n, 100.3),
                np.full(n, 99.7), np.full(n, 100.), np.full(n, 100.), np.full(n, .02),
                spec, "synthetic software fixture", True,
                {"storage": "UTC", "broker_clock": {"zone": "UTC", "valid_from": None, "valid_until": None}})


def pattern():
    bars = fixture()
    bars.high[-13] = 101.
    bars.low[-9] = 99.
    for offset, row in zip(range(-4, 0), [(100.,100.4,98.8,100.2),
                           (100.2,101.8,100.1,101.7), (101.7,101.9,101.2,101.6),
                           (101.3,101.7,100.9,101.5)]):
        for name, val in zip(("open", "high", "low", "close"), row): getattr(bars,name)[offset]=val
    return bars.validate()


class StructureTests(unittest.TestCase):
    def test_confirmed_pivots_do_not_appear_before_two_right_closes(self):
        _, _, _, s = analyze(pattern())
        self.assertIsNone(s[487]["swing_high"])
        self.assertIsNone(s[488]["swing_high"])
        self.assertEqual(s[489]["swing_high"], 101)
        self.assertEqual(s[493]["swing_low"], 99)

    def test_sweep_shift_fvg_requires_later_retest_for_both_sides(self):
        for side in (1, -1):
            b = pattern()
            if side < 0:
                b.open, b.close, b.high, b.low = 200-b.open, 200-b.close, 200-b.low, 200-b.high
            raw, atr, signals, structure = analyze(b)
            self.assertEqual(structure[496]["liquidity_sweep"], side)
            self.assertEqual(structure[497]["structure_break"], side)
            self.assertIsNone(signals[498])
            self.assertEqual(signals[499]["side"], side)
            self.assertEqual(signals[499]["kind"], "sweep_mss_fvg")
            self.assertTrue(np.isfinite(raw).all())
            self.assertEqual(raw.shape, (500, len(FEATURES)))
            self.assertTrue((atr > 0).all())

    def test_future_prices_never_repaint_features_or_signals(self):
        b = pattern()
        raw, atr, signals, structure = analyze(b)
        for end in (490, 494, 497, 499):
            prefix = replace(b, **{k:getattr(b,k)[:end].copy() for k in
                ("time","open","high","low","close","volume","spread")})
            a, r, g, s = analyze(prefix)
            np.testing.assert_array_equal(a, raw[:end])
            np.testing.assert_array_equal(r, atr[:end])
            self.assertEqual(g, signals[:end])
            self.assertEqual(s, structure[:end])

    def test_partial_m5_candle_does_not_change_bias(self):
        b = fixture()
        b.close[497], b.high[497] = 110, 110.1
        _,_,_,s = analyze(b)
        self.assertEqual(s[497]["m5_bias"], 0)
        self.assertEqual(s[498]["m5_bias"], 0)

    def test_missing_minute_cannot_form_three_candle_gap(self):
        b = pattern()
        b.time[-3:] += 60
        self.assertIsNone(analyze(b)[2][-1])

    def test_m5_is_not_silently_upsampled(self):
        with self.assertRaises(ValueError): analyze(synthetic_bars(800))


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"memory.sqlite"
        self.m=Memory(self.path)
        self.cfg=ScalpConfig()
        self.x=readout_context(np.arange(83)/83, {"side":1,"kind":"sweep_mss_fvg"})

    def tearDown(self):
        self.m.close(); self.tmp.cleanup()

    def learn(self, pid=10, net=-5):
        return learn_closed(self.m,"account-demo-profile",pid,"XAUUSD.m",self.x,net,10,100,200,202,self.cfg)

    def test_costed_reward_updates_once_and_survives_restart(self):
        self.assertTrue(self.learn())
        before=learning_state(self.m,"account-demo-profile")
        self.assertEqual(before["samples"],1)
        self.assertEqual(before["sum_r"],-.5)
        self.assertLess(entry_score(before,self.x,.68,self.cfg)[0],.68)
        self.m.close();self.m=Memory(self.path)
        self.assertFalse(self.learn())
        self.assertEqual(before,learning_state(self.m,"account-demo-profile"))

    def test_positive_reward_and_account_mode_separation(self):
        self.learn(net=15)
        self.assertGreater(entry_score(learning_state(self.m,"account-demo-profile"),self.x,.68,self.cfg)[0],.68)
        self.assertEqual(learning_state(self.m,"different-live-account")["samples"],0)

    def test_learning_is_bounded_and_does_not_change_risk(self):
        old=asdict(scalp_risk())
        for i in range(100): self.learn(pid=i,net=-1e9)
        s=learning_state(self.m,"account-demo-profile")
        self.assertLessEqual(np.linalg.norm(s["weights"]),self.cfg.max_weight_norm+1e-10)
        self.assertLessEqual(abs(entry_score(s,self.x,.68,self.cfg)[1]),.2)
        self.assertEqual(old,asdict(scalp_risk()))

    def test_future_outcome_rejected_and_atomic_failure_rolls_back_event(self):
        with self.assertRaises(ValueError):
            learn_closed(self.m,"x",1,"XAUUSD.m",self.x,2,1,100,300,200)
        self.m.put_state("scalp-learning:account-demo-profile",{"weights":[1e3]*READOUT_DIM})
        with self.assertRaises(ValueError): self.learn()
        self.assertEqual(self.m.db.execute("SELECT count(*) FROM events").fetchone()[0],0)


class M1TransportTests(unittest.TestCase):
    def test_m1_slot_allows_next_minute_but_not_duplicate_same_minute(self):
        api=FakeMT5();bridge=MT5(api);now=int(time.time()//300)*300+5
        with tempfile.TemporaryDirectory() as temp:
            memory=Memory(Path(temp)/"m.sqlite")
            try:
                for seconds,key in ((0,"first"),(60,"second")):
                    with patch("time.time",return_value=now+seconds):
                        api.quote.time_msc=(now+seconds)*1000
                        request=bridge.request("XAUUSD.m",1,1,fixture().spec,scalp_risk(),api.account_value)
                        result=bridge.send_order(request,key,memory,123,True,entry_period_seconds=60)
                        self.assertEqual(result["status"],"sent")
                        with self.assertRaises(Blocked):
                            bridge.send_order(request,key+"-other",memory,123,True,entry_period_seconds=60)
                self.assertEqual(api.calls,2)
            finally: memory.close()

    def test_fetch_uses_actual_m1_and_excludes_forming_bar(self):
        api=FakeMT5();api.TIMEFRAME_M1=1;api.TIMEFRAME_M5=5
        b=fixture(); raw=np.zeros(500,dtype=[(k,"f8") for k in
            ("time","open","high","low","close","tick_volume","spread")])
        for k in ("time","open","high","low","close"):raw[k]=getattr(b,k)
        raw["tick_volume"],raw["spread"]=100,2
        with patch.object(api,"copy_rates_from_pos",create=True,return_value=raw) as call:
            result=MT5(api).bars(b.spec,500)
            call.assert_called_once_with("XAUUSD.m",1,1,500)
        self.assertEqual(len(result.time),500)

    def test_structural_risk_size_and_too_wide_stop(self):
        api=FakeMT5();bridge=MT5(api);spec=fixture().spec
        for side,stop in ((1,1999.3),(-1,2001.0)):
            request=bridge.request(spec.symbol,side,1,spec,scalp_risk(),api.account_value,structural_stop=stop)
            loss=-api.order_calc_profit(request["type"],spec.symbol,request["volume"],request["price"],request["sl"])
            self.assertLessEqual(loss,api.account_value.equity*.001+1e-8)
        with self.assertRaises(Blocked): bridge.request(spec.symbol,1,1,spec,scalp_risk(),api.account_value,structural_stop=1990)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.m=Memory(self.root/"memory.sqlite");self.api=FakeMT5();self.bridge=MT5(self.api)
        self.now=int(time.time()//60)*60+5
        self.b=fixture();self.b.synthetic=False
        self.b.time=self.b.time-self.b.time[-1]+self.now-5-60
        self.api.quote=NS(bid=100.,ask=100.02,time_msc=self.now*1000)
        self.cfg=ScalpConfig();self.risk=scalp_risk();self.digest="a"*64
        self.meta={"strategy":VERSION,"spec":asdict(self.b.spec),"risk":asdict(self.risk),
            "scalp_config":asdict(self.cfg),"synthetic":False,"time_basis":self.b.time_basis}
        self.brain=NS(last_time=None,observe=lambda x,t:np.r_[x[-1],np.ones(64)])
        self.scope=learning_scope(self.api.account_value,self.digest)
        self.signal={"side":1,"kind":"sweep_mss_fvg","score":.7,"stop_bid":99.2,
            "gap_low":99.5,"gap_high":100.,"setup_id":"test-gap","formed_at":self.now-65}
        self.structure={"structure_break":0,"swing_high":100.5,"swing_low":99.2}
        self.news=self.root/"news.json"
        self.news.write_text(json.dumps({"coverage_start":iso(self.now-100),"coverage_end":iso(self.now+5000),
            "updated_at":iso(self.now-1),"source_url":"https://www.bls.gov/schedule/","events":[]}))

    def tearDown(self):self.m.close();self.tmp.cleanup()

    def run_cycle(self,execute=True,**kw):
        with patch("fxb.scalper_runtime.policy",return_value=(self.brain,self.meta,self.digest)), \
             patch.object(self.bridge,"bars",return_value=self.b), \
             patch("fxb.scalper_runtime.analyze",return_value=(np.zeros((500,19)),np.ones(500),[self.signal]*500,[self.structure]*500)), \
             patch("time.time",side_effect=lambda:self.now):
            return scalp_cycle(self.bridge,"model.npz",self.m,123,str(self.news),execute,
                               kill_file=str(self.root/"STOP"),**kw)

    def test_valid_setup_sends_once_with_500_context_and_attached_stops(self):
        r=self.run_cycle()
        self.assertEqual(r["status"],"sent")
        self.assertGreater(r["request"]["tp"],r["request"]["price"])
        self.assertLess(r["request"]["sl"],r["request"]["price"])
        self.assertEqual(self.run_cycle()["reason"],"already_processed_bar")
        self.assertEqual(self.api.calls,1)

    def test_dry_run_cannot_create_learning_samples_or_broker_orders(self):
        self.assertEqual(self.run_cycle(execute=False)["status"],"dry_run")
        self.assertEqual(self.api.calls,0)
        self.assertEqual(self.m.db.execute("SELECT count(*) FROM orders").fetchone()[0],0)
        self.assertEqual(learning_state(self.m,self.scope)["samples"],0)
        # Switching from observation to execution does not inherit its bar claim.
        self.assertEqual(self.run_cycle()["status"],"sent")

    def test_stale_signal_missing_news_and_manual_exposure_never_send(self):
        start,news=self.now,self.news.read_text()
        for which,reason in (("late","entry_window_expired"),("news","Invalid economic-calendar file"),
                             ("manual","account_has_exposure"),("spread","Spread is too large relative to ATR")):
            with self.subTest(which=which):
                self.now=start;self.news.write_text(news);self.api.open_positions=()
                self.api.quote.ask=100.02
                if which=="late":self.now+=25
                elif which=="news":self.news.unlink()
                elif which=="manual":self.api.open_positions=(NS(ticket=99,magic=0,symbol="XAUUSD.m"),)
                else:self.api.quote.ask=100.5
                self.assertEqual(self.run_cycle()["reason"],reason)
                self.assertEqual(self.api.calls,0)
                self.m.db.execute("DELETE FROM state WHERE key LIKE 'decision:%'");self.m.db.commit()

    def test_owned_time_exit_precedes_history_and_brain(self):
        self.api.open_positions=(NS(ticket=1,magic=MAGIC,symbol="XAUUSD.m",time=self.now-301,
            type=0,volume=.01,sl=99,tp=102),)
        self.news.unlink()
        with patch.object(self.bridge,"history_deals",side_effect=AssertionError("exit must not need history")):
            r=self.run_cycle()
        self.assertEqual(r["action"],"CLOSE")
        self.assertEqual(self.api.calls,1)

    def test_live_still_requires_matching_activation(self):
        self.api.account_value.trade_mode=2
        r=self.run_cycle(mode="live",expected_server="test")
        self.assertEqual(r["status"],"blocked")
        self.assertEqual(self.api.calls,0)

    def test_live_profile_can_send_when_explicit_activation_matches(self):
        self.api.account_value.trade_mode=2
        auth=LiveAuthorization.activate(123,"test",[self.digest],self.meta["risk"],123)
        # Set the test clock after the authorization's actual creation time.
        self.now=max(self.now,auth.issued_at+1)
        self.b.time=self.b.time-self.b.time[-1]+int(self.now//60)*60-60
        self.api.quote.time_msc=self.now*1000
        if self.now % 60 > 18:
            self.now=int(self.now//60)*60+65
            self.b.time+=60;self.api.quote.time_msc=self.now*1000
        result=self.run_cycle(mode="live",expected_server="test",authorization=auth)
        self.assertEqual(result["status"],"sent")
        self.assertEqual(self.api.calls,1)

    def test_cold_brain_latency_cannot_submit_a_late_signal(self):
        def slow_observe(x,t):
            self.now+=25
            return np.r_[x[-1],np.zeros(64)]
        self.brain.observe=slow_observe
        self.assertEqual(self.run_cycle()["reason"],"entry_window_expired")
        self.assertEqual(self.api.calls,0)

    def test_consecutive_loss_pause(self):
        self.m.put_state("scalp-learning:"+self.scope,{"loss_streak":3,"last_exit":self.now-100})
        self.assertEqual(self.run_cycle()["reason"],"consecutive_loss_pause")
        self.assertEqual(self.api.calls,0)

    def test_ambiguous_submission_cannot_be_resubmitted_after_restart(self):
        self.api.fail=TimeoutError()
        result=self.run_cycle()
        self.assertEqual(result["status"],"blocked")
        self.assertEqual(self.api.calls,1)
        self.m.close();self.m=Memory(self.root/"memory.sqlite")
        self.api.fail=None;self.now+=60;self.b.time+=60;self.api.quote.time_msc=self.now*1000
        self.assertEqual(self.run_cycle()["reason"],"uncertain_order_requires_reconciliation")
        self.assertEqual(self.api.calls,1)

    def test_broker_fee_reward_waits_for_full_close_and_is_not_duplicated(self):
        r=self.run_cycle();vol=r["volume"]
        entry=NS(ticket=34,order=12,position_id=77,symbol="XAUUSD.m",magic=MAGIC,entry=0,
            time=self.now,volume=vol,profit=0.,commission=-.2,swap=0.,fee=-.1,comment=r["request"]["comment"])
        close=NS(ticket=35,order=13,position_id=77,symbol="XAUUSD.m",magic=MAGIC,entry=1,
            time=self.now+60,volume=vol,profit=1.,commission=-.2,swap=-.1,fee=-.1,comment="tp")
        p=NS(ticket=88,identifier=77)
        self.assertEqual(settle_attempts(self.m,self.scope,"XAUUSD.m",[entry,close],[p],self.now+63,self.cfg),0)
        close.volume=vol/2
        self.assertEqual(settle_attempts(self.m,self.scope,"XAUUSD.m",[entry,close],[],self.now+63,self.cfg),0)
        close.volume=vol
        self.assertEqual(settle_attempts(self.m,self.scope,"XAUUSD.m",[entry,close],[],self.now+63,self.cfg),1)
        s=learning_state(self.m,self.scope)
        self.assertAlmostEqual(s["net_cashflow"],.3)
        self.assertEqual(settle_attempts(self.m,self.scope,"XAUUSD.m",[entry,close],[],self.now+64,self.cfg),0)
        self.assertEqual(s,learning_state(self.m,self.scope))

    def test_foreign_entry_or_manual_exit_is_never_a_learning_reward(self):
        r=self.run_cycle();vol=r["volume"]
        entry=NS(ticket=34,order=12,position_id=77,symbol="XAUUSD.m",magic=MAGIC,entry=0,
            time=self.now,volume=vol,profit=0.,commission=0.,swap=0.,fee=0.,comment=r["request"]["comment"])
        close=NS(ticket=35,order=13,position_id=77,symbol="XAUUSD.m",magic=0,entry=1,
            time=self.now+60,volume=vol,profit=100.,commission=0.,swap=0.,fee=0.,comment="manual")
        self.assertEqual(settle_attempts(self.m,self.scope,"XAUUSD.m",[entry,close],[],self.now+63,self.cfg),0)
        self.assertEqual(learning_state(self.m,self.scope)["samples"],0)


@unittest.skipUnless(os.environ.get("FXB_FULL_BRAIN_TEST")=="1","opt-in published graph integration")
class FullScalperTests(unittest.TestCase):
    def test_full_graph_m1_adapter_streams_and_profile_roundtrips(self):
        from fxb.fast_brain import FastWholeBrain
        b=pattern();raw,_,signals,_=analyze(b)
        brain=FastWholeBrain()
        encoded=brain.observe(raw,b.time)
        self.assertEqual(encoded.shape,(83,))
        self.assertGreater(brain.latest["spikes"],0)
        simulation=brain.latest["simulation_ms"]
        np.testing.assert_array_equal(brain.observe(raw,b.time),encoded)
        self.assertEqual(brain.latest["simulation_ms"],simulation)
        self.assertEqual(len(brain.ids),138639)
        with tempfile.TemporaryDirectory() as temp:
            b.synthetic=False
            bridge=MT5(FakeMT5())
            with patch.object(bridge,"spec",return_value=b.spec),patch.object(bridge,"bars",return_value=b):
                result=create_profile(bridge,b.spec.symbol,Path(temp)/"profile",0,0,0)
            loaded,meta,digest=policy(bridge,result["model"])
            self.assertEqual(meta["features"],FEATURES)
            self.assertEqual(meta["initial_policy"],"explicit_rules_with_zero_online_samples")
            self.assertIsNone(loaded.net)
            self.assertEqual(result["orders_sent"],0)


if __name__=="__main__":unittest.main()
