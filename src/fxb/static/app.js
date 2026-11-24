'use strict';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (n, digits = 0) => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const scalpReasons = {neural_warmup:'آماده‌سازی مغز با کندل‌های اخیر',neural_context_updated_waiting_for_fresh_ticks:'مغز به‌روز شد؛ انتظار تیک تازه',micro_waiting_for_quote_momentum:'انتظار حرکت تیک با پوشش اسپرد و هزینه‌ها',post_basket_cooldown:'مکث پس از بسته‌شدن سبد',fixed_lot_budget:'حجم انتخابی بالاتر از بودجهٔ ریسک یا مارجین',basket_budget:'ریسک یا مارجین کل سبد بالاتر از بودجه',target_cost_limit:'هزینه‌ها حد سود را از محدودهٔ اسکلپ خارج می‌کنند',worker_lock_busy:'یک موتور دیگر فعال است',worker_lock_legacy:'فرایند قبلی هنوز فعال است یا وضعیتش قابل تأیید نیست',no_confirmed_setup:'در انتظار فرصت معتبر',minimum_lot_budget:'حداقل لات بالاتر از بودجهٔ ریسک یا مارجین',entry_window_expired:'فرصت ورود منقضی شده',post_trade_cooldown:'مکث کوتاه پس از خروج',consecutive_loss_pause:'مکث پس از زیان‌های متوالی',daily_trade_limit:'سقف ورود روزانه',daily_loss:'حد زیان روزانه',maximum_drawdown:'حد افت سرمایه',account_has_exposure:'حساب موقعیت یا سفارش باز دارد',learned_entry_filter:'فیلتر آموخته‌شده ورود را رد کرد',entries_paused:'ورودهای جدید متوقف‌اند',momentum_breakout:'شکست مومنتوم M1',trend_pullback:'ادامهٔ روند پس از پولبک',live_momentum_breakout:'شکست زندهٔ قیمت',sweep_mss_fvg:'بازآزمایی FVG پس از Sweep',bos_fvg_retest:'بازآزمایی FVG پس از شکست'};
Object.assign(scalpReasons,{operator_stop:'توقف توسط کاربر',basket_risk_halt:'توقف ریسک سبد',basket_missing_protection:'حفاظت SL/TP ناقص',basket_holding_limit:'پایان زمان نگهداری سبد',favorable_quote_pressure_weakened:'محافظت سود پس از ضعف حرکت',basket_capacity:'ظرفیت قابل‌استفاده برای سفارش وجود ندارد',broker_volume_limit:'حد حجم بروکر'});
Object.assign(scalpReasons,{no_structure_setup:'تیک حرکت دارد؛ تأیید ساختار وجود ندارد',no_structure_target:'هدف نقدینگی معتبر پیدا نشد',structure_stop_distance:'فاصلهٔ ابطال ساختار نامعتبر است',structure_reward_risk:'هدف ساختاری نسبت سود به زیان خالص را پوشش نمی‌دهد',neural_and_structure_entry_accepted:'ورود پس از ارزیابی مغز و ساختار تأیید شد',basket_settled:'سبد تسویه شد؛ نتیجهٔ واقعی ثبت شد'});
const scalpReason = value => scalpReasons[value] || value || '—';
const structureDiagnostic = d => d?.stage==='structure_validation'
  ? ` · ${d.failure==='crossed'?'سطح ابطال پیش از ورود شکسته است':'حد ضرر ساختاری بیش از حد دور است'} · قیمت ${num(d.bid,3)} · SL ${num(d.stop_bid,3)} · فاصله ${num(d.distance_price,3)} (${num(d.distance_atr,2)} ATR) / حداکثر ${num(d.max_distance_price,3)}` : '';
const hash = new URLSearchParams(location.hash.slice(1));
if (hash.has('token')) { sessionStorage.setItem('fxb_token', hash.get('token')); history.replaceState(null, '', location.pathname); }
const token = sessionStorage.getItem('fxb_token') || '';
let demoReview=null, demoBusy=false, upgradeBusy=false;
let snapshot, currentPage = 'overview', initializedConfig = false, oldModels = '', oldDatasets = '', geometryLoaded = false, geometryLoading = false, lastJob = '', toastTimer;
const pageInfo = {
  overview:['OVERVIEW','مرکز فرماندهی معاملات','فعالیت مغز، وضعیت سرمایه و تصمیم‌های سیستم در یک محیط.'],
  brain:['NEURAL ENGINE','کانکتوم زنده','نورون‌های واقعی، دینامیک مدل اصلی و انتشار قابل مشاهدهٔ فعالیت.'],
  training:['GOLD SCALPER','اسکلپر طلا و آموزش','پروفایل M1 با زمینهٔ اخیر بازار و یادگیری از معاملات بسته‌شده.'],
  execution:['EXECUTION','اجرای کنترل‌شدهٔ معاملات','حساب مشخص، قواعد معلوم و کنترل ریسک مستقل از یادگیری مدل.'],
  journal:['MEMORY','حافظه و دفتر عملیات','تصمیم‌ها، نتیجهٔ اجرا و رویدادهای قابل پیگیری.'],
  settings:['CONFIGURATION','اتصال و تنظیمات','ترمینال محلی، دادهٔ زیستی و تقویم اقتصادی.']
};
function toast(message, error=false) { const el=$('toast');el.textContent=message;el.className=error?'error':'';el.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.hidden=true,error?9000:4500); }
async function api(path, body) {
  const response=await fetch('/api/'+path,{method:body===undefined?'GET':'POST',headers:{'Authorization':'Bearer '+token,...(body===undefined?{}:{'Content-Type':'application/json'})},body:body===undefined?undefined:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw Object.assign(new Error(data.error || data.detail || `HTTP ${response.status}`),{details:data.details,reason_code:data.reason_code});
  return data;
}
async function command(path, body={}) { try{const r=await api(path,body);toast(r.status==='started'?'عملیات آغاز شد.':r.status==='flatten_requested'?'درخواست بستن ثبت شد؛ تأیید بروکر را بررسی کنید.':'عملیات ثبت شد.');await poll();return r;}catch(e){toast(e.message,true);throw e;} }
function page(name) {if(!pageInfo[name])return;currentPage=name;document.querySelectorAll('[data-view]').forEach(el=>el.classList.toggle('active',el.dataset.view.split(' ').includes(name)));document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('active',el.dataset.page===name));const info=pageInfo[name];$('page-label').textContent=info[0];$('page-title').textContent=info[1];$('page-subtitle').textContent=info[2];$('financial-strip').hidden=!['overview','execution'].includes(name);if(name==='journal')loadJournal();if(brain)brain.resize();}
document.querySelectorAll('[data-page]').forEach(el=>el.addEventListener('click',()=>page(el.dataset.page)));
document.querySelectorAll('[data-go]').forEach(el=>el.addEventListener('click',()=>page(el.dataset.go)));
document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();page('overview');});
document.querySelectorAll('[data-action]').forEach(el=>el.addEventListener('click',async()=>{el.disabled=true;try{await command(el.dataset.action);}catch{}finally{el.disabled=false;}}));
function table(headers,rows,empty){return rows.length?`<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`:`<div class="empty">${esc(empty)}</div>`;}
const kabTime=t=>new Date(t*1000).toLocaleTimeString('en-GB',{timeZone:'Asia/Kabul',hour12:false});
function render(s){
  snapshot=s;$('version').textContent='v'+s.version;$('preview-badge').hidden=!s.preview;
  $('connection').textContent=s.connected?`MT5 · ${s.account?.mode?.toUpperCase()||'CONNECTED'}`:'MT5 DISCONNECTED';$('connection').className='badge '+(s.connected?'green':'');
  const a=s.account;['equity','balance','profit'].forEach(k=>$(k).textContent=a?num(a[k],2):'—');$('profit').className=a?(a.profit>=0?'positive':'negative'):'';
  $('account-label').textContent=a?`#${a.login} · ${a.server}`:'حساب متصل نشده';$('currency').textContent=a?`${a.currency} · updated ${kabTime(a.updated_at)} Kabul`:'دادهٔ مستقیم ترمینال';
  const demoEligible=!s.preview&&s.connected&&a?.mode==='demo';
  $('demo-session-panel').hidden=!demoEligible;
  const demoUnavailable=!demoEligible||s.running||s.job.status==='running'||demoBusy;
  $('demo-session-preview').disabled=demoUnavailable;
  if(demoReview&&(!demoEligible||a.login!==demoReview.login||a.server!==demoReview.server||a.equity!==demoReview.equity||a.balance!==demoReview.balance)){demoReview=null;$('demo-session-note').textContent='حساب یا سرمایه تغییر کرد؛ بررسی تازه بگیرید.';}
  $('demo-session-review').hidden=!demoReview;
  $('demo-session-restart').disabled=demoUnavailable||!demoReview;
  const worker=s.worker||{}, neural=s.brain.activity.execution||{};
  const initializing=neural.phase==='warming_up'&&neural.remaining_bars==null;
  const enginePhase=worker.phase==='failed'?'FAILED':worker.phase==='blocked'?'BLOCKED':!s.running?'STANDBY':worker.phase==='stopping'?'STOPPING':neural.phase==='submitting'?'SUBMITTING':neural.phase==='managing_positions'?'MANAGING':neural.phase==='checking_order'?'CHECKING':!s.entries_enabled?'PAUSED':neural.phase==='entry_blocked'?'BLOCKED':neural.phase==='warming_up'?(initializing?'INITIALIZING':'WARMUP'):worker.phase==='starting'?'STARTING':worker.phase==='waiting'?'WAITING':'RUNNING';
  $('engine-state').textContent=enginePhase;
  $('engine-detail').textContent=s.running?`${s.mode.toUpperCase()} · ${scalpReason(worker.reason||neural.reason||'')}${worker.cycle_completed_at?' · '+kabTime(worker.cycle_completed_at)+' کابل':''}`:(worker.reason||'منتظر اتصال و انتخاب مدل');
  if(s.running)$('engine-detail').textContent+=structureDiagnostic(worker.details||s.scalper?.last_decision?.details||neural.details);
  $('run-badge').textContent=s.running?s.mode.toUpperCase():'STOPPED';
  const error=s.error||(s.job.status==='failed'?s.job.error:null);$('system-alert').hidden=!error;$('system-alert').textContent=error||'';
  const manifest=s.brain.manifest;
  $('brain-state').textContent=manifest?'MODEL LOADED':'MODEL NOT INSTALLED';$('brain-state').className='badge '+(manifest?'green':'amber');
  if(manifest){$('nodes').textContent=num(manifest.neurons);$('edges').textContent=num(manifest.connections);$('synapses').textContent=num(manifest.synapse_contacts);$('neuron-count').textContent=num(manifest.neurons)+' NEURONS';if(!geometryLoaded&&!geometryLoading)loadGeometry();}
  else{$('brain-empty').textContent='برای دریافت مدل اصلی به تنظیمات بروید.';$('brain-empty').hidden=false;}
  const activity=s.job.status==='running'&&s.job.kind==='train'&&s.job.progress.activity?s.job.progress.activity:s.brain.activity;
  const processed=Number(activity.simulation_ms)>0, observation=activity.observation;
  $('spikes').textContent=num(processed?activity.spikes:null);$('active-neurons').textContent=num(processed?activity.active_neurons:null);$('simulation-ms').textContent=num(processed?activity.simulation_ms:null,0)+' ms';$('wall-seconds').textContent=activity.wall_seconds==null?'—':num(activity.wall_seconds,2)+' s';
  $('activity-label').textContent=s.running&&(observation?.remaining_bars>0||neural.phase==='warming_up')?'WARMUP':processed?(s.running&&activity.observed_at?'M1 PROCESSED':'LAST WINDOW'):'AWAITING INPUT';
  let source=activity.input_source==='closed_market_bars'?'فعالیت آخرین پنجرهٔ واقعی؛ ورودی مغز با کندل بسته‌شدهٔ M1 به‌روز می‌شود.':activity.spikes?'خروجی آخرین شبیه‌سازی واقعی · تحریک دستی':'شبیه‌ساز هنوز ورودی را پردازش نکرده است.';
  if(neural.phase==='warming_up')source=initializing?'در حال بارگذاری و آماده‌سازی گراف؛ هنوز آمادهٔ استنتاج نیست.':`آماده‌سازی مرحله‌ای؛ ${num(neural.remaining_bars)} کندل باقی مانده. صف سفارش بین مراحل بررسی می‌شود.`;
  else if(neural.phase==='managing_positions')source+=' مدیریت خروج سبد در اولویت است.';
  else if(neural.reason&&s.running)source+=' '+scalpReason(neural.reason)+'.';
  if(activity.observed_at)source+=' آخرین پردازش: '+kabTime(activity.observed_at)+' کابل.';
  if(observation)source+=` از شروع اجرا ${num(observation.processed_bars)} کندل پردازش شده؛ مجموع ${num(observation.recent_spikes)} شلیک در ${num(observation.recent_windows)} پنجرهٔ اخیر.`;
  if(processed&&activity.spikes===0)source+=' در آخرین پنجره شلیکی ثبت نشده؛ این عدد به‌تنهایی به معنای توقف پردازش نیست.';
  $('activity-source').textContent=source;
  renderPipeline(s,activity);
  drawActivity(activity.pool_activity||[]);if(brain)brain.activity(activity.indices||[],activity.counts||[]);
  $('stimulate-button').disabled=!!s.running||s.job.status==='running'||!manifest;
  $('decision-count').textContent=s.decisions.length+' EVENTS';
  if(s.decisions.length){$('decision-stream').className='';$('decision-stream').innerHTML=s.decisions.slice(-4).reverse().map(d=>`<div class="decision-row"><span class="badge ${d.action==='BUY'?'green':d.action==='SELL'?'amber':''}">${esc(d.action)}</span><div>${esc(d.model)}<small>${esc(d.reason||d.status)}</small></div><time>${kabTime(d.time)}</time></div>`).join('');}
  const checks=[['کانکتوم کامل',!!manifest,manifest?'138,639 neurons · full graph':'نصب از تنظیمات'],['ترمینال معاملاتی',s.connected,s.connected?`${a.login} / ${a.mode}`:'Windows MT5 required'],['مدل با دادهٔ واقعی',s.models.some(m=>!m.synthetic),`${s.models.filter(m=>!m.synthetic).length} profiles`],['تقویم ورود',s.news.ready,s.news.ready?'Fresh coverage':'نیاز به تقویم معتبر']];
  $('readiness').innerHTML=checks.map(([name,ok,detail])=>`<div>${esc(name)}<b class="${ok?'ok':''}">${ok?'READY':'PENDING'}</b><small>${esc(detail)}</small></div>`).join('');
  const modelKey=JSON.stringify(s.models);if(modelKey!==oldModels){oldModels=modelKey;const checked=new Set([...document.querySelectorAll('[name=models]:checked')].map(el=>el.value));$('models-table').innerHTML=table(['MODEL / SYMBOL','ENCODER','SOURCE','TRADES','WIN RATE','RETURN','MAX DD','PROFIT FACTOR'],s.models.map(m=>{const h=m.holdout||{};return[esc(m.name)+'<br>'+esc(m.symbol),esc(m.encoder),`<span class="badge ${m.synthetic?'amber':'green'}">${m.synthetic?'SYNTHETIC':m.strategy?'ONLINE · NO HOLDOUT':'MARKET DATA'}</span>`,num(h.trades),h.win_rate==null?'—':num(h.win_rate*100,1)+'%',h.return_pct==null?'—':num(h.return_pct,2)+'%',h.max_drawdown_pct==null?'—':num(h.max_drawdown_pct,2)+'%',num(h.profit_factor,2)];}),'هنوز مدلی آموزش داده نشده است. نتایج پس از آموزش نمایش داده می‌شوند.');
    $('model-choices').innerHTML=s.models.length?s.models.map(m=>`<label class="model-choice"><input type="checkbox" name="models" value="${esc(m.path)}" ${checked.has(m.path)?'checked':''}><span>${esc(m.symbol)} / ${esc(m.name)}<small>${esc(m.encoder)} · ${m.synthetic?'SYNTHETIC — dry run only':m.strategy?`M1 ${(m.scalp_style||'structure').toUpperCase()} · ${num(m.risk.fraction*100,2)}% risk · ${m.risk.max_trades===0?'∞':num(m.risk.max_trades)}/day${m.micro_config?.sizing_mode==='capacity'?' · AUTO':''}${m.micro_config?.exit_policy==='confirmed'?' · PATIENT':''}${m.micro_config?.entry_policy==='structure'?' · STRUCTURE':''}${m.micro_config?.lot_mode==='risk'?' · RISK LOT':''}`:'REAL MARKET DATA'}</small></span></label>`).join(''):'مدل آموزش‌دیده‌ای موجود نیست.';}
  $('micro-upgrade-button').disabled=upgradeBusy||s.running||s.preview||s.job.status==='running';
  const scalp=s.scalper;
  if(scalp){const l=scalp.learning||{},t=scalp.structure||{},d=scalp.last_decision||{},r=scalp.risk||{},stats=scalp.activity_stats||{};
    $('scalp-status').className='pad';$('scalp-status').innerHTML=table(['SYMBOL','SWING HIGH / LOW','M5 BIAS','LEARNED BASKETS / TRADES','REALIZED NET','LAST DECISION'],[[esc(scalp.symbol),num(t.swing_high,2)+' / '+num(t.swing_low,2),esc(t.m5_bias??'—'),num(l.samples),num(l.net_cashflow,2)+' '+esc(a?.currency||''),esc(d.action||'WAIT')+' · '+esc(scalpReason(d.reason_code||d.reason))]],'');
    $('scalp-limits').textContent=`${(scalp.style||'structure').toUpperCase()} · ریسک هر معامله ${num(r.fraction*100,2)}٪ (${num(scalp.risk_budget_cash,2)} ${a?.currency||''}) · حد زیان روزانه ${num(r.daily_loss*100,1)}٪ · ورود امروز ${num(scalp.entries_today)} / ${r.max_trades===0?'بدون سقف تعداد':num(r.max_trades)} · ${scalp.style==='micro'?`یک سبد هم‌زمان · ${scalp.micro_config?.sizing_mode==='capacity'?'تعداد خودکار':num(scalp.micro_config?.orders)} × ${scalp.micro_config?.lot_mode==='risk'?'حجم از بودجهٔ ریسک':num(scalp.micro_config?.lot,2)+' لات'} · ریسک کل سبد ${num((scalp.micro_config?.basket_risk||0)*100,2)}٪`:'یک موقعیت هم‌زمان'} · امتیاز ورود احتمال برد نیست.`;
    const reasons=Object.entries(stats.reasons||{}).sort((x,y)=>y[1]-x[1]).slice(0,5).map(([why,count])=>`${scalpReason(why)}: ${num(count)}`).join(' · ');
    let detail=`کندل‌های بررسی‌شدهٔ امروز: ${num(stats.bars||0)} · کندل‌های دارای فرصت: ${num(stats.signal_bars||0)}. ${reasons}`;
    if(scalp.entry_gate)detail+=` | مانع فعلی ورود: ${scalpReason(scalp.entry_gate)}. سقف سرمایهٔ ثبت‌شده ${num(scalp.risk_peak_equity,2)} و سرمایهٔ ابتدای روز ${num(scalp.day_start_equity,2)} ${a?.currency||''}.`;
    if(d.sizing&&d.sizing.minimum_lot!=null){const z=d.sizing;detail+=` | حداقل لات ${num(z.minimum_lot,2)}: زیان محاسبه‌شده تا حد ضرر و هزینه‌ها ${num(z.minimum_lot_risk_cash,2)} ${z.currency} (${num(z.minimum_lot_risk_percent,2)}٪ سرمایه)، بودجهٔ انتخاب‌شده ${num(z.risk_budget_cash,2)} ${z.currency}. مارجین لازم ${num(z.minimum_lot_margin_cash,2)} / بودجهٔ مارجین ${num(z.margin_budget_cash,2)} ${z.currency}.`;}
    if(scalp.style==='micro'){detail+=` | فشار تیک: ${num(scalp.pressure?.velocity,3)} / ثانیه · پوزیشن‌های سبد: ${num(d.open_positions)} · خالص تخمینی خروج: ${num(d.estimated_liquidation_net,2)} ${a?.currency||''}`;} detail+=' | آخرین پاداش: '+num(l.last_reward_r,3)+'R · میانگین پاداش: '+num(l.samples?l.sum_r/l.samples:null,3)+'R · نسخهٔ یادگیری: '+num(l.revision)+' · تغییر وزن: '+num(l.last_weight_delta,6); if(d.exit_diagnostics){const x=d.exit_diagnostics;detail+=' | محافظت سود: '+(x.armed?'آماده':'در انتظار سود معنادار')+' · سود لازم '+num(x.arm_net_cash,2)+' · اوج سود '+num(x.peak_net_cash,2)+' · برگشت لازم '+num(x.required_giveback_cash,2);} const plan=d.plan||scalp.basket?.plan;if(plan?.count_limits)detail+=' | ظرفیت محاسبه‌شده: '+num(plan.requested_orders)+' · عامل محدودکننده: '+plan.limiting_factors.join(', '); if(d.details)detail+=' | '+JSON.stringify(d.details); if(d.error)detail+=' | '+JSON.stringify(d.error); $('scalp-diagnostics').textContent=detail;
  }else{$('scalp-limits').textContent='حدود ریسک از پروفایل فعال خوانده می‌شود.';$('scalp-diagnostics').textContent='';}
  const manual=s.manual_order, pending=manual&&['queued','processing'].includes(manual.status);
  const m1=s.active_models.length===1&&s.models.some(m=>m.path===s.active_models[0]&&m.strategy);
  $('manual-mode').textContent=s.running?`${s.mode.toUpperCase()} · #${a?.login||''}`:'STOPPED';
  $('manual-check').disabled=manualBusy||pending||!s.running||!m1||s.preview||initializing||worker.phase==='stopping';
  $('manual-send').disabled=$('manual-check').disabled||s.mode==='dry-run';
  $('manual-summary').textContent=`حساب ${a?.login||'—'} · ${s.mode.toUpperCase()} · حجم کل انتخاب‌شده ${(Number($('manual-form').elements.lot.value)*Number($('manual-form').elements.count.value)).toFixed(2)} لات. بررسی، سفارش باز نمی‌کند؛ ارسال دستی سفارش واقعی به همین حساب می‌فرستد.`;
  if(manual)$('manual-result').textContent=JSON.stringify(manual,null,2);
  $('manual-status').textContent=manual?`${manual.status.toUpperCase()} · ${scalpReason(manual.reason_code||manual.error?.code||manual.error?.reason||manual.reason||'')}${manual.broker?' · MT5 '+manual.broker.retcode+' · '+manual.broker.comment:''}`:'بررسی اتصال: ابتدا یک سفارش ۰٫۰۱ لات را با «بررسی بدون ارسال» بررسی کنید.';
  if(manual)$('manual-status').textContent+=structureDiagnostic(manual.details||manual.error?.details);
  if(manual?.plan)$('manual-status').textContent+=` | SL ${num(manual.plan.sl,3)} · TP ${num(manual.plan.tp,3)} · نسبت سود به زیان خالص ${num(manual.plan.net_reward_risk,2)} · ${manual.plan.protection_source==='confirmed_structure'?'حدود ساختاری':'حدود نوسان و بروکر'}`;
  if(manual?.reason_code==='fixed_lot_budget'&&manual.details){const z=manual.details;$('manual-status').textContent+=` | ریسک سفارش ${num(z.risk_cash,2)} / بودجه ${num(z.risk_budget_cash,2)} ${z.currency}.`;if(z.minimum_equity_for_selected_risk!=null)$('manual-status').textContent+=` ریسک کل انتخاب‌شده ${num(z.total_risk_cash,2)}؛ حداقل سرمایهٔ محاسباتی با همین حدود ${num(z.minimum_equity_for_selected_risk,2)} ${z.currency}، با فرض رفع توقف‌های قبلی. این برآورد با قیمت تغییر می‌کند.`;}

  const market=scalp?.market_data;
  $('market-data-status').textContent=market?`دادهٔ M1: ${market.status} · بسته‌شدن: ${new Date(market.bar_close_utc).toLocaleString('en-GB',{timeZone:'Asia/Kabul',hour12:false})} کابل · سن هنگام بررسی: ${num(market.age_seconds,2)} ثانیه`:'در انتظار دریافت کندل M1.';
  const sf=$('scalp-form');
  if(!sf.dataset.costsSet){const previous=s.datasets.filter(d=>!d.synthetic&&d.symbol===sf.elements.symbol.value&&d.spec).at(-1);if(previous){for(const [field,key] of [['commission','commission_per_lot_side'],['slippage','slippage_points'],['overnight','overnight_cost_per_lot']]){if(!sf.elements[field].value)sf.elements[field].value=previous.spec[key];}sf.dataset.costsSet='1';}}
  $('scalp-prepare-button').disabled=s.running||!s.connected||s.preview||s.job.status==='running';
  const datasetKey=JSON.stringify(s.datasets);if(datasetKey!==oldDatasets){oldDatasets=datasetKey;const old=$('dataset-select').value;$('dataset-select').innerHTML='<option value="">دیتاست واقعی را انتخاب کنید</option>'+s.datasets.filter(d=>!d.synthetic).map(d=>`<option value="${esc(d.path)}">${esc(d.symbol+' / '+d.path)}</option>`).join('');$('dataset-select').value=old;}
  $('positions-table').className='table-scroll';$('positions-table').innerHTML=table(['TICKET','SYMBOL','SIDE','LOTS','ENTRY','SL / TP','P&L','OWNER'],s.positions.map(p=>[esc(p.ticket),esc(p.symbol),p.type===0?'BUY':'SELL',num(p.volume,2),num(p.price_open,2),num(p.sl,2)+' / '+num(p.tp,2),num(p.profit,2),p.owned?'FXB':'MAGIC '+esc(p.magic)]),s.connected?'موقعیت بازی در حساب وجود ندارد.':'اتصال به MT5 برقرار نیست.');
  if(!initializedConfig){for(const[k,id]of[['login','login-input'],['server','server-input'],['terminal','terminal-input']])$(id).value=s.config[k]||'';const c=s.config.broker_clock||{};$('broker-zone-input').value=c.zone||'UTC';$('clock-from-input').value=c.valid_from||'';$('clock-until-input').value=c.valid_until||'';initializedConfig=true;}
  $('news-status').textContent=s.news.reason;
  const refresh=s.news.refresh||{};
  $('calendar-auto-button').disabled=s.preview||s.news.automatic;
  $('calendar-auto-status').textContent=s.news.automatic?`تقویم خودکار فعال · ${refresh.status||'pending'}${refresh.reason?' · '+refresh.reason:''}`:'تقویم خودکار خاموش؛ با فعال‌سازی، هر ساعت هنگام بازبودن FXB بازخوانی می‌شود.';
  $('live-status').hidden=!s.live_activation;if(s.live_activation)$('live-status').textContent=`Account ${s.live_activation.login} · ${s.live_activation.active?'ACTIVE':'EXPIRED'} · expires ${kabTime(s.live_activation.expires_at)} Kabul`;
  const job=s.job,progress=job.progress||{};$('job-panel').hidden=job.status!=='running';$('job-title').textContent=(job.kind||'').toUpperCase();$('job-detail').textContent=[progress.stage,progress.completed?`${num(progress.completed)} / ${num(progress.total)} bars`:null,progress.epoch?`epoch ${progress.epoch}/${progress.epochs}`:null,progress.frame?`${progress.frame}/${progress.frames} windows`:null,progress.bytes?`${num(progress.bytes/1e6,1)} MB`:null,progress.message].filter(Boolean).join(' · ');
  const jobKey=job.started_at+':'+job.status;if(lastJob&&jobKey!==lastJob&&['completed','failed','cancelled'].includes(job.status))toast(job.status==='failed'?job.error:job.status==='cancelled'?'عملیات لغو شد.':'عملیات کامل شد.',job.status==='failed');lastJob=jobKey;
  $('start-button').disabled=s.running||!s.connected||s.preview||s.job.status==='running';
}
let pollBusy=false;
async function poll(){if(pollBusy)return;pollBusy=true;try{render(await api('status'));}catch(e){$('system-alert').textContent=e.message;$('system-alert').hidden=false;}finally{pollBusy=false;}}
async function loadJournal(){try{const rows=await api('journal');$('journal-table').className='table-scroll';$('journal-table').innerHTML=table(['KABUL TIME','SYMBOL','KIND','DETAIL'],rows.map(r=>[kabTime(r.occurred_at),esc(r.symbol),esc(r.kind),esc(JSON.stringify(r.payload).slice(0,700))]),'هنوز رویداد معاملاتی ثبت نشده است.');}catch(e){toast(e.message,true);}}
$('refresh-journal').addEventListener('click',loadJournal);
function formHandler(id,handler){$(id).addEventListener('submit',async e=>{e.preventDefault();const data=Object.fromEntries(new FormData(e.target));const b=e.target.querySelector('[type=submit]');b.disabled=true;try{await handler(data);}catch{}finally{b.disabled=false;}});}
formHandler('stimulate-form',d=>command('brain/stimulate',{channels:d.channels,frames:Number(d.frames),rate:Number(d.rate)}));
formHandler('connect-form',async d=>{await command('account/configure',{login:Number(d.login),server:d.server.trim(),terminal:d.terminal.trim(),broker_clock:{zone:d.broker_zone.trim()||'UTC',valid_from:d.clock_from.trim()||null,valid_until:d.clock_until.trim()||null}});await command('account/connect');});
formHandler('export-form',d=>command('data/export',{symbol:d.symbol.trim(),bars:Number(d.bars),commission:Number(d.commission),slippage:Number(d.slippage),overnight:Number(d.overnight)}));
formHandler('scalp-form',d=>command('scalper/prepare',{symbol:d.symbol.trim(),commission:Number(d.commission),slippage:Number(d.slippage),overnight:Number(d.overnight),style:d.style,...(d.style!=='structure'?{risk_percent:Number(d.risk_percent),max_trades:Number(d.max_trades),daily_loss_percent:Number(d.daily_loss_percent)}:{}),...(d.style==='micro'?{basket_orders:Number(d.basket_orders||5),basket_lot:Number(d.basket_lot),lot_mode:d.lot_mode,basket_risk_percent:Number(d.basket_risk_percent),sizing_mode:d.sizing_mode,hold_minutes:Number(d.hold_minutes),min_hold_seconds:Number(d.min_hold_seconds),confirmation_seconds:Number(d.confirmation_seconds)}:{})}));
function scalpFields(){
  const f=$('scalp-form'),style=$('scalp-style').value,micro=style==='micro';
  for(const name of ['risk_percent','max_trades','daily_loss_percent'])f.elements[name].disabled=style==='structure';
  const count=f.elements.max_trades;count.min=micro?'0':'1';if(micro)count.removeAttribute('max');else count.max='250';
  if(f.dataset.previousStyle&&f.dataset.previousStyle!==style)count.value=micro?'0':style==='active'?'250':'40';
  f.dataset.previousStyle=style;
  $('micro-fields').hidden=!micro;$('micro-exit-fields').hidden=!micro;
  for(const name of ['lot_mode','sizing_mode','basket_orders','basket_lot','basket_risk_percent','hold_minutes','min_hold_seconds','confirmation_seconds'])f.elements[name].disabled=!micro;
  f.elements.basket_orders.disabled=!micro||f.elements.sizing_mode.value==='capacity';
}
$('scalp-style').addEventListener('change',scalpFields);$('scalp-form').elements.sizing_mode.addEventListener('change',scalpFields);scalpFields();
$('micro-upgrade-button').addEventListener('click',async()=>{
  const chosen=[...document.querySelectorAll('[name=models]:checked')].map(el=>el.value);
  if(chosen.length!==1||!snapshot.models.some(m=>m.path===chosen[0]&&m.scalp_style==='micro')){toast('یک پروفایل Micro را برای ادامهٔ همان یادگیری انتخاب کنید.',true);return;}
  upgradeBusy=true;$('micro-upgrade-button').disabled=true;
  try{const r=await command('scalper/upgrade',{model:chosen[0]});
    document.querySelectorAll('[name=models]').forEach(el=>el.checked=el.value===r.model);
    $('micro-upgrade-note').textContent='ارتقا انجام شد و مدل جدید انتخاب شد. تجربه‌های همان حساب حفظ شده‌اند. موتور خاموش است؛ حالت اجرا را انتخاب و شروع کنید. مدل: '+r.model;
  }catch{}finally{upgradeBusy=false;await poll();}
});
formHandler('train-form',d=>command('training/start',{dataset:d.dataset,epochs:Number(d.epochs),encoder:d.encoder}));
formHandler('start-form',d=>command('execution/start',{models:[...document.querySelectorAll('[name=models]:checked')].map(el=>el.value),mode:d.mode,hours:Number(d.hours||8),acknowledged_login:d.acknowledged_login?Number(d.acknowledged_login):null}));
$('mode-select').addEventListener('change',()=>{$('live-fields').hidden=$('mode-select').value!=='live';});
$('flatten-button').addEventListener('click',()=>$('confirm-dialog').showModal());
$('cancel-flatten').addEventListener('click',()=>$('confirm-dialog').close());
$('confirm-flatten').addEventListener('click',async()=>{$('confirm-dialog').close();try{await command('execution/flatten');}catch{}});
$('save-calendar').addEventListener('click',async()=>{try{const f=$('calendar-file').files[0];if(!f)throw Error('فایل تقویم را انتخاب کنید.');if(f.size>2e6)throw Error('فایل بیش از حد بزرگ است.');await command('calendar/save',JSON.parse(await f.text()));}catch(e){toast(e.message,true);}});
$('close-inspector').addEventListener('click',()=>$('inspect-dialog').close());
function drawActivity(values){const c=$('activity-chart'),r=c.getBoundingClientRect();if(!r.width)return;const dpr=Math.min(devicePixelRatio,2);c.width=r.width*dpr;c.height=r.height*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const max=Math.max(...values,.05);for(let i=0;i<64;i++){const h=Math.max(1,(values[i]||0)/max*(r.height-10));ctx.fillStyle=values[i]?'#9ddbb6':'#253c40';ctx.fillRect(i*r.width/64,r.height-h,r.width/64-1.5,h);}}

// Every rendered position is an actual FlyWire coordinate. Camera rotation is visual only.
class BrainView{
  constructor(canvas,data){this.canvas=canvas;this.data=data;this.positions=new Float32Array(data.positions);this.indexMap=new Map(data.indices.map((v,i)=>[v,i]));this.n=data.indices.length;this.values=new Float32Array(this.n);this.yaw=.13;this.pitch=-.15;this.zoom=1;this.rotating=!matchMedia('(prefers-reduced-motion: reduce)').matches;this.gl=canvas.getContext('webgl',{alpha:true,antialias:true,preserveDrawingBuffer:false});if(this.gl)this.setupGL();else this.ctx=canvas.getContext('2d');this.events();this.resize();this.frame();}
  setupGL(){const gl=this.gl;const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    const vs=compile(gl.VERTEX_SHADER,`attribute vec3 aPosition;attribute float aActivity;uniform vec2 uRotation;uniform float uAspect;uniform float uZoom;uniform float uDpr;varying float vActivity;varying float vDepth;void main(){vec3 p=vec3(aPosition.x,-aPosition.z,aPosition.y);float cy=cos(uRotation.x),sy=sin(uRotation.x),cx=cos(uRotation.y),sx=sin(uRotation.y);p=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);p=vec3(p.x,p.y*cx-p.z*sx,p.y*sx+p.z*cx);float perspective=2.6/(2.6+p.z);gl_Position=vec4(p.x*perspective*uZoom/uAspect,p.y*perspective*uZoom,p.z*.2,1.);gl_PointSize=(aActivity>0.?3.3:1.25)*uDpr;vActivity=aActivity;vDepth=p.z;}`);
    const fs=compile(gl.FRAGMENT_SHADER,`precision mediump float;varying float vActivity;varying float vDepth;void main(){float d=length(gl_PointCoord-.5);if(d>.5)discard;vec3 color=vActivity>0.?vec3(.84,1.,.54):mix(vec3(.22,.48,.55),vec3(.56,.83,.75),clamp(.6-vDepth,0.,1.));float alpha=(vActivity>0.?.88:.22)*(1.-smoothstep(.16,.5,d));gl_FragColor=vec4(color,alpha);}`);
    const program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);this.program=program;
    const position=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,position);gl.bufferData(gl.ARRAY_BUFFER,this.positions,gl.STATIC_DRAW);const loc=gl.getAttribLocation(program,'aPosition');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);
    this.activityBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.activityBuffer);gl.bufferData(gl.ARRAY_BUFFER,this.values,gl.DYNAMIC_DRAW);const al=gl.getAttribLocation(program,'aActivity');gl.enableVertexAttribArray(al);gl.vertexAttribPointer(al,1,gl.FLOAT,false,0,0);
    this.uniforms=Object.fromEntries(['uRotation','uAspect','uZoom','uDpr'].map(k=>[k,gl.getUniformLocation(program,k)]));gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);gl.clearColor(0,0,0,0);
  }
  resize(){const r=this.canvas.getBoundingClientRect();if(!r.width)return;this.width=r.width;this.height=r.height;this.dpr=Math.min(devicePixelRatio||1,2);this.canvas.width=r.width*this.dpr;this.canvas.height=r.height*this.dpr;if(this.gl)this.gl.viewport(0,0,this.canvas.width,this.canvas.height);}
  activity(indices,counts){const key=indices.join(',')+':'+counts.join(',');if(key===this.activityKey)return;this.activityKey=key;this.values.fill(0);indices.forEach((index,i)=>{const n=this.indexMap.get(index);if(n!==undefined)this.values[n]=Math.min(counts[i]||1,10);});if(this.gl){this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.activityBuffer);this.gl.bufferSubData(this.gl.ARRAY_BUFFER,0,this.values);}}
  project(i){const p=this.positions;let x=p[i*3],y=-p[i*3+2],z=p[i*3+1];const a=x*Math.cos(this.yaw)+z*Math.sin(this.yaw);z=-x*Math.sin(this.yaw)+z*Math.cos(this.yaw);x=a;const b=y*Math.cos(this.pitch)-z*Math.sin(this.pitch);z=y*Math.sin(this.pitch)+z*Math.cos(this.pitch);y=b;const t=2.6/(2.6+z)*this.zoom;return[(x*t/ (this.width/this.height)+1)*this.width/2,(1-y*t)*this.height/2];}
  frame(){if(this.width&&this.canvas.offsetParent!==null){if(this.rotating&&!this.drag)this.yaw+=.0006;if(this.gl){const g=this.gl;g.clear(g.COLOR_BUFFER_BIT);g.useProgram(this.program);g.uniform2f(this.uniforms.uRotation,this.yaw,this.pitch);g.uniform1f(this.uniforms.uAspect,this.width/this.height);g.uniform1f(this.uniforms.uZoom,this.zoom*1.35);g.uniform1f(this.uniforms.uDpr,this.dpr);g.drawArrays(g.POINTS,0,this.n);}else{const c=this.ctx;c.setTransform(this.dpr,0,0,this.dpr,0,0);c.clearRect(0,0,this.width,this.height);for(let i=0;i<this.n;i+=4){const[x,y]=this.project(i);c.fillStyle=this.values[i]?'#d7fda0':'#5c9d9e55';c.fillRect(x,y,this.values[i]?3:1, this.values[i]?3:1);}}}requestAnimationFrame(()=>this.frame());}
  events(){this.canvas.addEventListener('pointerdown',e=>{this.drag=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);});this.canvas.addEventListener('pointermove',e=>{if(this.drag){this.yaw+=(e.clientX-this.drag[0])*.006;this.pitch+=(e.clientY-this.drag[1])*.006;this.drag=[e.clientX,e.clientY];}});this.canvas.addEventListener('pointerup',()=>this.drag=null);this.canvas.addEventListener('pointercancel',()=>this.drag=null);this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(.4,Math.min(3,this.zoom*Math.exp(-e.deltaY*.001)));},{passive:false});this.canvas.addEventListener('dblclick',async e=>{this.rotating=false;const r=this.canvas.getBoundingClientRect();let best=-1,distance=200;const zoom=this.zoom;this.zoom*=this.gl?1.35:1;for(let i=0;i<this.n;i++){const[x,y]=this.project(i),d=(x-e.clientX+r.left)**2+(y-e.clientY+r.top)**2;if(d<distance){best=i;distance=d;}}this.zoom=zoom;if(best>=0)inspectNeuron(this.data.indices[best]);});new ResizeObserver(()=>this.resize()).observe(this.canvas);}
}
let brain;
async function loadGeometry(){geometryLoading=true;try{const data=await api('brain/geometry');brain=new BrainView($('brain-canvas'),data);geometryLoaded=true;$('brain-empty').hidden=true;}catch(e){$('brain-empty').textContent='نمای مغز در دسترس نیست: '+e.message;}finally{geometryLoading=false;}}
async function inspectNeuron(index){$('neuron-detail').innerHTML='<div class="empty">در حال خواندن اتصال‌های واقعی…</div>';$('inspect-dialog').showModal();try{const d=await api('brain/neuron/'+index);$('neuron-detail').innerHTML=`<p>ROOT ID: <b>${esc(d.root_id)}</b> · INDEX ${d.index} · DEGREE ${num(d.degree)}</p><p dir="rtl">حداکثر ۳۰ اتصال با بیشترین وزن؛ علامت وزن از مدل اصلی گرفته شده است.</p>`+table(['PRESYNAPTIC ID','POSTSYNAPTIC ID','SIGNED WEIGHT'],d.connections.map(c=>[esc(c.pre),esc(c.post),esc(c.weight)]),'اتصالی یافت نشد.');}catch(e){$('neuron-detail').textContent=e.message;}}
$('orbit').addEventListener('click',()=>{if(brain){brain.rotating=!brain.rotating;$('orbit').textContent=brain.rotating?'Ⅱ':'▷';}});
$('reset-view').addEventListener('click',()=>{if(brain){brain.yaw=.13;brain.pitch=-.15;brain.zoom=1;}});
$('fit-view').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.brain-stage').requestFullscreen();}catch(e){toast(e.message,true);}});
function clock(){$('clock').textContent=new Date().toLocaleTimeString('en-GB',{timeZone:'Asia/Kabul',hour12:false})+' KABUL';}clock();setInterval(clock,1000);poll();setInterval(poll,1500);

let manualBusy=false, manualRetry=null;
async function manualRequest(checkOnly){
  if(manualBusy)return;
  manualBusy=true;$('manual-check').disabled=true;$('manual-send').disabled=true;
  const f=$('manual-form').elements;
  const payload={side:Number(f.side.value),lot:Number(f.lot.value),count:Number(f.count.value),check_only:checkOnly};
  const signature=JSON.stringify(payload);
  if(!manualRetry||manualRetry.signature!==signature)manualRetry={signature,id:crypto.randomUUID()};
  try{const r=await api('execution/manual',{...payload,request_id:manualRetry.id});$('manual-result').textContent=JSON.stringify(r,null,2);manualRetry=null;await poll();}
  catch(e){$('manual-result').textContent=e.message+(e.details?'\n'+JSON.stringify(e.details,null,2):'')+' · Repeating the same request retains its ID.';toast(e.message,true);}
  finally{manualBusy=false;await poll();}
}
$('manual-check').addEventListener('click',()=>manualRequest(true));
$('manual-form').addEventListener('submit',e=>{e.preventDefault();manualRequest(false);});

$('demo-session-preview').addEventListener('click',async()=>{
  demoBusy=true;$('demo-session-preview').disabled=true;
  try{
    demoReview=await api('execution/demo-session/preview',{});
    const r=demoReview,b=r.previous,n=r.proposed;
    $('demo-session-account').textContent=`حساب دمو #${r.login} · ${r.server} · موجودی ${num(r.balance,2)} ${r.currency} · سرمایه ${num(r.equity,2)} ${r.currency}`;
    $('demo-session-values').innerHTML=table(['مبنا','قبلی','دورهٔ جدید'],[
      ['سقف سرمایه',num(b.peak||0,2),num(n.peak,2)],['سرمایهٔ ابتدای دوره',num(b.day_start||0,2),num(n.day_start,2)],['ورود امروز',num(b.entries||0),num(n.entries)]],'');
    $('demo-session-restart').textContent=`ثبت مبنای ${num(r.equity,2)} ${r.currency} برای حساب #${r.login}`;
    $('demo-session-note').textContent='مقادیر را بررسی کنید؛ تأیید تا دو دقیقه معتبر است. این کار مجوز عبور از بودجهٔ حجم یا ریسک نیست.';
  }catch(e){toast(e.message,true);$('demo-session-note').textContent=e.message;}
  finally{demoBusy=false;await poll();}
});
$('demo-session-restart').addEventListener('click',async()=>{
  if(!demoReview||demoBusy)return;
  demoBusy=true;$('demo-session-restart').disabled=true;
  try{
    const r=await api('execution/demo-session/restart',{token:demoReview.token,acknowledged_login:demoReview.login});
    demoReview=null;
    $('demo-session-note').textContent=`دورهٔ تازه برای حساب #${r.login} با سرمایهٔ ${num(r.equity,2)} ${r.currency} ثبت شد. موتور خاموش است؛ پس از شروع، بررسی سفارش جدید بگیرید.`;
    toast('دورهٔ دمو ثبت شد؛ هیچ سفارشی ارسال نشد.');
  }catch(e){if(e.reason_code==='demo_session_changed')demoReview=null;toast(e.message,true);$('demo-session-note').textContent=e.message;}
  finally{demoBusy=false;await poll();}
});

const featureLabels = ['return_1_atr','return_3_atr','body_atr','upper_wick_atr','lower_wick_atr','range_atr','distance_swing_high_atr','distance_swing_low_atr','swing_trend','structure_break','liquidity_sweep','displacement','fvg_side','distance_fvg_atr','fvg_width_atr','spread_atr','relative_tick_volume','closed_m5_bias','premium_discount'];
function renderPipeline(s,a){
  const p=s.pipeline||{}, d=p.last_evaluation, entry=p.last_entry, l=s.scalper?.learning||{}, cloud=s.cloud||{};
  $('pipeline-summary').textContent=`کندل بسته‌شدهٔ M1 ← ۱۹ ویژگی ← ۱۵۲ کانال تحریک ← کانکتوم ← خوانش یادگیرنده + تیک تازه + ساختار قیمت ← کنترل ریسک ← سفارش. ${a.market_bar_open_at?'آخرین کندل: '+kabTime(a.market_bar_open_at+60)+' کابل. ':''}${entry?'آخرین ورود '+(entry.source==='manual'?'دستی':'خودکار')+': '+kabTime(entry.evaluated_at)+' کابل.':'هنوز ورود ثبت نشده است.'}`;
  $('decision-attribution').innerHTML=d?table(['DECISION','BASE / FINAL / MIN','MARKET R','NEURAL R','TICK R','REVISION'],[[esc(d.id?.slice(0,8)),num(d.base_score,3)+' / '+num(d.entry_score,3)+' / '+num(d.threshold,3),num(d.market_contribution_r,4),num(d.neural_contribution_r,4),num(d.tick_contribution_r,4),num(d.learning_revision)]],''):'';
  const plan=entry?.plan;
  if(plan)$('decision-attribution').innerHTML+=table(['LAST ENTRY SL / TP','NET REWARD / RISK','LOT × COUNT','STOP RISK / MARGIN'],[[num(plan.sl,3)+' / '+num(plan.tp,3),num(plan.net_reward_risk,2),num(plan.lot,4)+' × '+num(plan.requested_orders),num(plan.total_risk_cash,2)+' / '+num(plan.total_margin_cash,2)]],'');
  $('reward-summary').textContent=l.samples?`سبدهای آموخته‌شده: ${num(l.samples)} · برد ${num(100*l.wins/l.samples,1)}٪ · نتیجهٔ خالص ${num(l.net_cashflow,2)} ${s.account?.currency||''} · آخرین پاداش/تنبیه ${num(l.last_reward_r,3)}R · خطای پیش‌بینی ${num(l.last_reward_error_r,3)}R · تغییر واقعی وزن Δ=${num(l.last_weight_delta,6)} · نسخهٔ یادگیری ${num(l.revision)}. خوانش خروجی آموزش می‌بیند؛ اتصال‌های زیستی ثابت‌اند.`:'هنوز نتیجهٔ تسویه‌شده‌ای برای یادگیری ثبت نشده؛ وزن‌های خوانش آنلاین پس از بسته‌شدن کامل سبد تغییر می‌کنند.';
  if(l.performance_samples)$('reward-summary').textContent+=` نسبت مجموع سود به زیان ${l.gross_loss>0?num(l.gross_profit/l.gross_loss,2):'—'} در ${num(l.performance_samples)} سبد ثبت‌شده از نسخهٔ 0.7.`;
  $('neural-inputs').innerHTML=table(['FEATURE','NORMALIZED VALUE','POSITIVE Hz / NEGATIVE Hz'],featureLabels.map((name,i)=>[name,num(a.input_features?.[i],4),num(a.input_rates?.[i*4],2)+' / '+num(a.input_rates?.[(i+19)*4],2)]),'');
  const states={not_configured:'تنظیم نشده',configured:'در انتظار بررسی',connected:'متصل',syncing:'در حال ارسال',retry_pending:'ارسال ناموفق؛ تلاش خودکار بعدی'};
  $('cloud-status').textContent=`Supabase: ${states[cloud.status]||'تنظیم نشده'} · ${cloud.project_url||''} · رویدادهای در صف ${num(s.journal.pending_sync)}${cloud.last_success?' · آخرین تأیید '+kabTime(cloud.last_success)+' کابل':''}`;
  $('cloud-form').querySelector('button').disabled=s.preview;
  drawRaster(a);
}
function drawRaster(a){
  const canvas=$('spike-raster'),w=canvas.clientWidth;
  if(!w)return;
  const h=180,dpr=Math.min(devicePixelRatio,2);canvas.width=w*dpr;canvas.height=h*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#8babb1';ctx.font='11px monospace';
  const r=a.raster, ids=[...new Set(r?.indices||[])];
  ctx.fillText('0 ms',5,175);ctx.fillText(num(a.window_ms||20,1)+' ms',w-70,175);
  if(r){const rows=new Map(ids.map((id,i)=>[id,i]));ctx.fillStyle='#b2e78a';r.indices.forEach((id,i)=>{const x=5+r.times_ms[i]/(a.window_ms||20)*(w-15),y=8+rows.get(id)/Math.max(ids.length,1)*145;ctx.fillRect(x,y,2,Math.max(2,Math.min(8,135/Math.max(ids.length,1))));});}
  $('raster-note').textContent=r?`هر نقطه یک شلیک ثبت‌شده در شبیه‌سازی است؛ هر ردیف یک نورون فعال. ${num(r.indices.length)} شلیک در آخرین پنجرهٔ ${num(a.window_ms,1)} ms شبیه‌سازی${r.truncated?'؛ نمایش به ۴۰۹۶ شلیک اول محدود شده':''}. این نمودار با پردازش واقعی به‌روز می‌شود.`:'زمان تک‌تک شلیک‌ها هنوز در این اجرا ثبت نشده است؛ نمودار ساختگی نمایش داده نمی‌شود.';
}
formHandler('cloud-form',async()=>{const file=$('cloud-file').files[0];if(!file||file.size>8192){toast('فایل اتصال معتبر انتخاب کنید.',true);return;}try{await command('memory/connect',JSON.parse(await file.text()));$('cloud-file').value='';}catch(e){toast(e.message,true);}});
