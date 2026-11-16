/* LatticeIQ dashboard. Reads data/index.json, data/stocks/<T>.json and data/track.json
   (written by `python -m scripts.export_dashboard`). No build step. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const cfg = window.LATTICE || {};
  const domains = window.LATTICE_DOMAINS || {};
  const LWC = window.LightweightCharts;
  const state = { index: null, track: null, sel: null, h: "20", view: "overview", sort: "az", filter: "all", f: { sec: "", move: "", rank: "", range: "" }, period: "3M", stock: null, stars: new Set(), hit: 0, today: { quotes: {}, state: null, fetched: null }, ctype: "line", vol: true };
  const cache = new Map();
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PERIOD = { "1W": 5, "1M": 22, "3M": 66, "6M": 132, "1Y": 252 };
  const PERIOD_WORDS = { "1W": "past week", "1M": "past month", "3M": "past 3 months", "6M": "past 6 months", "1Y": "past year" };
  const PLAIN = { "50": "Right half the time", "80": "Right 4 times in 5", "90": "Right 9 times in 10" };

  // ---------- today's prices so far (from the local server while the US market is open) ----------
  const quote = (t) => { const q = state.today.quotes[t]; return q && state.index && q.date > state.index.asof && q.prev > 0 ? q : null; };
  const P = (s) => { const q = quote(s.t); return q ? q.last : s.price; };
  const C = (s) => { const q = quote(s.t); return q ? q.last / q.prev - 1 : s.chg; };
  const clock = (iso) => new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  function whenLocal(iso) {
    const d = new Date(iso), now = new Date();
    return (d.toDateString() === now.toDateString() ? "today" : "on " + d.toLocaleDateString("en-GB", { weekday: "long" })) + " at " + clock(iso);
  }
  function marketLine() {
    const st = state.today.state, ix = state.index;
    if (!st) return "";
    const n = ix.stocks.filter((s) => quote(s.t)).length;
    if (n) return `${st.open ? "The US market is open." : "The US market has closed for today."} Today's prices are as of ${clock(state.today.fetched)} your time and can be delayed by up to 15 minutes.`;
    if (st.open) return "The US market is open. Today's prices are loading; they appear within a few minutes.";
    return `The US market is closed and opens ${whenLocal(st.next_open)} your time. Until then the latest prices are the close of ${day(ix.asof, true)}.`;
  }

  // ---------- small helpers ----------
  const money = (v) => "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const signed = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%";
  const pill = (v, d = 2) => `<span class="pill ${v >= 0 ? "up" : "down"}">${signed(v, d)}</span>`;
  const date = (s) => new Date(s + "T12:00:00");
  const day = (s, long) => date(s).toLocaleDateString("en-GB", long ? { weekday: "short", day: "numeric", month: "short" } : { day: "numeric", month: "short" });
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const big = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : String(n);
  const hue = (t) => { let h = 0; for (const c of t) h = (h * 31 + c.charCodeAt(0)) % 360; return h; };
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } } };

  function spark(vals, w, h) {
    const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
    const pts = vals.map((v, i) => `${(i / (vals.length - 1) * w).toFixed(1)},${(h - 1 - (v - lo) / span * (h - 2)).toFixed(1)}`).join(" ");
    return `<svg class="spark ${vals[vals.length - 1] >= vals[0] ? "up" : "down"}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}"/></svg>`;
  }
  const logoTag = (t, size) => `<span class="logo${size ? " " + size : ""}" data-logo="${t}" style="--hue:${hue(t)}"></span>`;
  const darkNow = () => (document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  // -> { url, kind }: kind "clear" = known see-through logo, null = find out from the image once it has loaded
  function logoSource(t) {
    if (cfg.logoDevToken) return { kind: "clear", url: "https://img.logo.dev/ticker/" + encodeURIComponent(t.replace("-", ".")) + "?token=" + encodeURIComponent(cfg.logoDevToken) +
      "&size=96&retina=true&format=png&theme=" + (darkNow() ? "dark" : "light") + "&fallback=404" };
    if (cfg.useTickerLogos !== false && domains[t]) return { kind: null, url: "https://cdn.tickerlogos.com/" + domains[t] };
    return null;
  }
  // Does the picture have a see-through edge? Solid squares fill the tile; see-through ones sit on a light tile.
  const logoKinds = new Map();
  function logoKind(img) {
    try {
      const c = document.createElement("canvas"); c.width = c.height = 24;
      const x = c.getContext("2d", { willReadFrequently: true });
      x.drawImage(img, 0, 0, 24, 24);
      const p = x.getImageData(0, 0, 24, 24).data;
      let clear = 0;
      for (let i = 0; i < 24; i++) for (const k of [i, 23 * 24 + i, i * 24, i * 24 + 23]) if (p[k * 4 + 3] < 128) clear++;
      return clear > 6 ? "clear" : "full";
    } catch (e) { return "clear"; }                       // the picture could not be read: the safe choice
  }
  function paintLogos(root) {
    root.querySelectorAll("[data-logo]").forEach((el) => {
      const t = el.dataset.logo;
      el.removeAttribute("data-logo");
      el.classList.remove("full", "clear");
      el.style.setProperty("--hue", hue(t));
      el.textContent = t.replace(/[^A-Z]/g, "").slice(0, 2);
      const src = logoSource(t);
      if (!src) return;
      const img = new Image();                       // sits over the letters and shows only once it has loaded
      img.alt = ""; img.loading = "lazy"; img.decoding = "async";
      if (!src.kind) img.crossOrigin = "anonymous";  // lets the page look at the edge pixels (see logoKind)
      img.onload = () => {
        let kind = src.kind || logoKinds.get(src.url);
        if (!kind) { kind = logoKind(img); logoKinds.set(src.url, kind); }
        el.classList.add(kind);
        img.classList.add("ok");
      };
      img.onerror = () => img.remove();
      img.src = src.url;
      el.append(img);
    });
  }

  // ---------- the forecast cone, drawn on the chart ----------
  class Cone {
    constructor() { this.d = null; this.p = 1; this.views = ["bottom", "top"].map((z) => ({ zOrder: () => z, renderer: () => ({ draw: (t) => this.draw(t, z) }) })); }   // cone under the candles, labels over the grid
    attached(p) { this.chart = p.chart; this.series = p.series; this.request = p.requestUpdate; }
    detached() { this.chart = null; }
    paneViews() { return this.views; }
    autoscaleInfo() { return this.d ? { priceRange: { minValue: this.d.bands["90"][0], maxValue: this.d.bands["90"][1] } } : null; }
    set(d, animate) {
      this.d = d;
      if (reduceMotion || !d || !animate) { this.p = 1; this.request && this.request(); return; }
      const t0 = performance.now();
      const step = (now) => {
        if (this.d !== d) return;                          // a newer range replaced this one: leave its progress alone
        this.p = Math.min(1, (now - t0) / 520);
        this.request && this.request();
        if (this.p < 1 && this.d === d) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
    draw(target, layer) {
      const d = this.d;
      if (!d || !this.chart) return;
      const ts = this.chart.timeScale();
      const x = (time) => ts.timeToCoordinate(time);
      const y = (price) => this.series.priceToCoordinate(price);
      target.useMediaCoordinateSpace(({ context: c, mediaSize }) => {
        const x0 = x(d.last);
        if (x0 === null) return;
        const n = Math.max(1, Math.round(d.h * (1 - Math.pow(1 - this.p, 3))));       // ease out
        const xs = [x0];
        for (let i = 1; i <= n; i++) { const v = x(d.future[i - 1]); if (v === null) break; xs.push(v); }
        const edge = (lv, sign) => xs.map((_, i) => y(d.price * Math.exp(sign * Math.log(d.bands[lv][1] / d.price) * Math.sqrt(i / d.h))));
        c.save();
        if (layer === "bottom") for (const lv of ["90", "80", "50"]) {
          const up = edge(lv, 1), dn = edge(lv, -1);
          c.beginPath();
          xs.forEach((v, i) => (i ? c.lineTo(v, up[i]) : c.moveTo(v, up[i])));
          for (let i = xs.length - 1; i >= 0; i--) c.lineTo(xs[i], dn[i]);
          c.closePath();
          c.fillStyle = d.colors[lv];
          c.fill();
        }
        if (layer === "bottom") {
          c.strokeStyle = d.colors.rule; c.lineWidth = 1;
          c.beginPath(); c.moveTo(x0, 0); c.lineTo(x0, mediaSize.height); c.stroke();
          if (d.earnAt !== null && this.p > 0.4) {
            const xe = x(d.future[d.earnAt]);
            if (xe !== null) {
              c.strokeStyle = d.colors.amber; c.setLineDash([4, 4]); c.lineWidth = 1.5;
              c.beginPath(); c.moveTo(xe, 24); c.lineTo(xe, mediaSize.height); c.stroke(); c.setLineDash([]);
            }
          }
          c.restore();
          return;
        }
        c.font = "12px Archivo, system-ui, sans-serif"; c.textBaseline = "top";
        const label = (text, lx, ly, align, color) => {          // text on a patch of the card colour, so grid lines do not cross it
          const w = c.measureText(text).width, left = align === "right" ? lx - w : lx;
          c.fillStyle = d.colors.sheet; c.fillRect(left - 3, ly - 2, w + 6, 17);
          c.fillStyle = color; c.textAlign = align; c.fillText(text, lx, ly);
        };
        label(d.label || "last close", x0 - 6, 6, "right", d.colors.text);
        if (d.earnAt !== null && this.p > 0.4) {
          const xe = x(d.future[d.earnAt]);
          if (xe !== null) {
            const left = xe <= mediaSize.width - 118;
            label("earnings expected", xe + (left ? 6 : -6), left && xe - x0 > 8 ? 6 : 26, left ? "left" : "right", d.colors.amberText);
          }
        }
        if (this.p === 1 && xs.length > 1) {
          const xe = xs[xs.length - 1] + 6;
          c.fillStyle = d.colors.text; c.textAlign = "left"; c.textBaseline = "middle";
          for (const lv of ["90", "80", "50"]) c.fillText(lv + "%", xe, y(d.bands[lv][1]) + 7);
        }
        c.restore();
      });
    }
  }

  // ---------- ranges that follow today's price ----------
  // The range made at the last close, moved to the price now and sized for the trading time left to the SAME end
  // date: half-width x sqrt(days left / days in the window). With no price today it is the range as made.
  function sessionDone() {                                 // share of today's trading session that is over, 0 to 1
    const st = state.today.state;
    return st && st.open && st.closes ? Math.min(1, Math.max(0, 1 - (new Date(st.closes) - Date.now()) / (6.5 * 3600e3))) : 1;
  }
  function rangeNow(s, hh) {
    const r = s.ranges[hh], H = Number(hh), q = quote(s.t), at = q ? s.future.indexOf(q.date) : -1;
    const made = { end: r.end, earn: r.earn, bands: r.bands, half: r.half, live: false, anchor: s.price, anchorTime: s.candles[s.candles.length - 1][0], from: 0, steps: H };
    if (at < 0 || at >= H) return made;
    const k = Math.sqrt(Math.max(0.25, H - at - sessionDone()) / H), bands = {}, half = {};
    for (const lv of ["50", "80", "90"]) {
      const hw = Math.log(1 + r.half[lv]) * k;
      bands[lv] = [q.last * Math.exp(-hw), q.last * Math.exp(hw)];
      half[lv] = Math.exp(hw) - 1;
    }
    return { end: r.end, earn: r.earn, bands, half, live: true, anchor: q.last, anchorTime: q.date, from: at + 1, steps: Math.max(1, H - 1 - at), made };
  }

  // ---------- chart ----------
  let chart, chartBox, area, candles, volume, cone, tip, marks, marksC, timeIx = new Map();
  function sizeChart() {                                   // true once the chart has a real size
    const w = chartBox.clientWidth, h = chartBox.clientHeight;
    if (!w || !h) return false;
    chart.resize(w, h);
    return true;
  }
  // show the chosen history plus the look-ahead, with room on the right for the cone's labels
  function setRange() {
    const s = state.stock, H = Number(state.h);
    if (!s || !sizeChart()) return;
    const show = Math.min(s.candles.length, PERIOD[state.period]);
    const perBar = Math.max(1, (chartBox.clientWidth - 70) / (show + H));
    const pad = Math.ceil(46 / perBar);
    chart.timeScale().setVisibleRange({ from: s.candles[s.candles.length - show][0], to: s.future[Math.min(s.future.length - 1, H - 1 + pad)] });
  }
  function chartColors() {
    return { text: css("--ink-2"), rule: css("--ink-3"), ink: css("--ink"), grid: css("--rule-soft"), up: css("--up"), down: css("--down"),
      sheet: css("--sheet"), "90": css("--cone-90"), "80": css("--cone-80"), "50": css("--cone-50"), amber: css("--amber-fill"), amberText: css("--amber") };
  }
  function buildChart() {
    const k = chartColors();
    // The chart is sized by hand, not by the library: while its page is hidden the box is 0 wide, and a date
    // range set before the library noticed the new size left the candles squeezed into a corner.
    chartBox = document.createElement("div");
    chartBox.style.cssText = "width:100%;height:100%";
    tip = document.createElement("div");
    tip.className = "tip"; tip.hidden = true;
    $("chart").append(chartBox, tip);
    chart = LWC.createChart(chartBox, {
      width: chartBox.clientWidth || 600, height: chartBox.clientHeight || 400,
      layout: { background: { color: "transparent" }, textColor: k.text, fontFamily: "Archivo, system-ui, sans-serif", fontSize: 12, attributionLogo: true },
      grid: { vertLines: { visible: false }, horzLines: { color: k.grid, style: LWC.LineStyle.Dotted } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, fixLeftEdge: true },
      crosshair: { mode: LWC.CrosshairMode.Magnet, vertLine: { color: k.rule, labelVisible: true, style: LWC.LineStyle.Dashed, width: 1, labelBackgroundColor: k.ink },
        horzLine: { color: k.rule, style: LWC.LineStyle.Dotted, width: 1, labelBackgroundColor: k.ink } },
      handleScale: { axisPressedMouseMove: { price: false } },
    });
    // The line is always there (it carries the price scale and the cone); in candle view it is see-through.
    area = chart.addSeries(LWC.AreaSeries, { lineWidth: 2, priceLineStyle: LWC.LineStyle.Dotted, priceLineWidth: 1, crosshairMarkerRadius: 5, crosshairMarkerBorderWidth: 2 });
    marks = LWC.createSeriesMarkers(area, []);
    area.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.2 } });
    candles = chart.addSeries(LWC.CandlestickSeries, { borderVisible: false, priceLineVisible: false, lastValueVisible: false });
    marksC = LWC.createSeriesMarkers(candles, []);
    volume = chart.addSeries(LWC.HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "vol", priceLineVisible: false, lastValueVisible: false });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.88, bottom: 0 } });
    cone = new Cone();
    area.attachPrimitive(cone);
    let lastW = 0;
    new ResizeObserver(() => {
      const w = chartBox.clientWidth;
      if (!sizeChart() || w === lastW) return;
      lastW = w;
      setRange();
    }).observe(chartBox);
    chart.subscribeCrosshairMove(showTip);
  }
  // the floating label: a past day's price and change, today so far, or the likely range on a future day
  function showTip(p) {
    const s = state.stock;
    if (!s || !p.time || !p.point) { tip.hidden = true; return; }
    const q = quote(s.t), i = timeIx.get(p.time);
    let html = "";
    if (i !== undefined) {
      const c = s.candles[i], prev = i ? s.candles[i - 1][4] : null;
      html = `<b>${day(c[0], true)} ${c[0].slice(0, 4)}</b><span class="big">${money(c[4])}</span>${prev ? pill(c[4] / prev - 1) : ""}` +
        (state.ctype === "candles" ? `<span class="sub">open ${c[1].toFixed(2)}, high ${c[2].toFixed(2)}, low ${c[3].toFixed(2)}</span>` : "");
    } else if (q && p.time === q.date) {
      html = `<b>Today so far</b><span class="big">${money(q.last)}</span>${pill(q.last / q.prev - 1)}` +
        (state.ctype === "candles" ? `<span class="sub">open ${q.open.toFixed(2)}, high ${q.high.toFixed(2)}, low ${q.low.toFixed(2)}</span>` : "");
    } else {
      const r = rangeNow(s, state.h), n = s.future.indexOf(p.time) + 1 - r.from;
      if (n < 1 || n > r.steps) { tip.hidden = true; return; }
      const hw = Math.log(r.bands["80"][1] / r.anchor) * Math.sqrt(n / r.steps);
      html = `<b>${day(p.time, true)}</b><span class="big">${money(r.anchor * Math.exp(-hw))} to ${money(r.anchor * Math.exp(hw))}</span>` +
        `<span class="sub">likely range in ${n} trading day${n > 1 ? "s" : ""}, right 4 times in 5</span>`;
    }
    tip.innerHTML = html;
    tip.hidden = false;
    const w = tip.offsetWidth, W = chartBox.clientWidth;
    tip.style.left = Math.max(4, p.point.x > W / 2 ? p.point.x - w - 16 : Math.min(p.point.x + 16, W - w - 64)) + "px";   // beside the pointer, never over it
  }
  // colours and which series shows: a line with a soft fill (green if the period is up, red if down), or candles
  function styleSeries() {
    const s = state.stock, k = chartColors(), line = state.ctype !== "candles", clear = "rgba(0,0,0,0)";
    let c = k.up;
    if (s) {
      const show = Math.min(s.candles.length, PERIOD[state.period]), q = quote(s.t);
      c = (q ? q.last : s.price) >= s.candles[s.candles.length - show][4] ? k.up : k.down;
    }
    const st = state.today.state;
    if (s) renderPeriod(k);
    volume.applyOptions({ visible: state.vol });
    area.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: state.vol ? 0.2 : 0.06 } });
    area.applyOptions({ priceLineVisible: line, priceLineColor: c, lineColor: line ? c : clear, topColor: line ? c + "33" : clear, bottomColor: line ? c + "00" : clear,
      crosshairMarkerVisible: line, crosshairMarkerBackgroundColor: c, crosshairMarkerBorderColor: k.sheet, lastValueVisible: line,
      lastPriceAnimation: line && st && st.open && !reduceMotion ? LWC.LastPriceAnimationMode.Continuous : LWC.LastPriceAnimationMode.Disabled });
    candles.applyOptions({ visible: !line, lastValueVisible: !line, priceLineVisible: !line, priceLineStyle: LWC.LineStyle.Dotted, priceLineWidth: 1, upColor: k.up, downColor: k.down, wickUpColor: k.up, wickDownColor: k.down });
    chart.applyOptions({ layout: { textColor: k.text }, grid: { horzLines: { color: k.grid } },
      crosshair: { vertLine: { color: k.rule, labelBackgroundColor: k.ink }, horzLine: { color: k.rule, labelBackgroundColor: k.ink } } });
  }
  // the line above the chart (change, highest and lowest close of the chosen period) and the marks on the chart:
  // the period's high and low, and an "E" on each past earnings day
  function renderPeriod(k) {
    const s = state.stock, q = quote(s.t), show = Math.min(s.candles.length, PERIOD[state.period]);
    const part = s.candles.slice(-show), now = q ? q.last : s.price, first = part[0][4];
    let hi = part[0], lo = part[0];
    for (const c of part) { if (c[4] > hi[4]) hi = c; if (c[4] < lo[4]) lo = c; }
    $("perf").innerHTML = `${pill(now / first - 1)}<span>${PERIOD_WORDS[state.period]}</span><span>high <b>${money(Math.max(hi[4], now))}</b></span><span>low <b>${money(Math.min(lo[4], now))}</b></span>`;
    const m = [];
    if (show > 5) {
      m.push({ time: hi[0], position: "aboveBar", color: k.text, shape: "arrowDown", text: "High " + hi[4].toFixed(2), size: 0.6 });
      if (lo[0] !== hi[0]) m.push({ time: lo[0], position: "belowBar", color: k.text, shape: "arrowUp", text: "Low " + lo[4].toFixed(2), size: 0.6 });
    }
    for (const e of s.earnings.history) {                    // the first trading day on or after each report
      const c = s.candles.find((x) => x[0] >= e.date);
      if (c && c[0] !== hi[0] && c[0] !== lo[0]) m.push({ time: c[0], position: "belowBar", color: k.amber, shape: "circle", text: "E", size: 0.7 });
    }
    m.sort((a, b) => a.time.localeCompare(b.time));
    const line = state.ctype !== "candles";                  // marks sit on whichever series is showing
    marks.setMarkers(line ? m : []);
    marksC.setMarkers(line ? [] : m);
  }
  function themeChart() {
    if (!chart) return;
    if (state.stock) drawStock(false, true); else styleSeries();
  }
  // history, then today's price so far (if the market has traded today), then empty days for the cone
  function setSeries() {
    const s = state.stock, k = chartColors(), q = quote(s.t), at = q ? s.future.indexOf(q.date) : -1;
    timeIx = new Map(s.candles.map((c, i) => [c[0], i]));
    area.setData(s.candles.map((c) => ({ time: c[0], value: c[4] })).concat(s.future.map((t, i) => (i === at ? { time: t, value: q.last } : { time: t }))));
    candles.setData(s.candles.map((c) => ({ time: c[0], open: c[1], high: c[2], low: c[3], close: c[4] }))
      .concat(at >= 0 ? [{ time: q.date, open: q.open, high: q.high, low: q.low, close: q.last }] : []));
    volume.setData(s.candles.map((c) => ({ time: c[0], value: c[5], color: (c[4] >= c[1] ? k.up : k.down) + "3d" }))
      .concat(at >= 0 ? [{ time: q.date, value: q.volume, color: (q.last >= q.open ? k.up : k.down) + "3d" }] : []));
    styleSeries();
  }
  function coneData() {
    const s = state.stock, r = rangeNow(s, state.h), fut = s.future.slice(r.from);
    let earnAt = null;
    if (r.earn && s.earnings.next) {
      const i = fut.findIndex((t) => t >= s.earnings.next);
      earnAt = i < 0 || i > r.steps + 2 ? null : Math.min(r.steps - 1, i);
    }
    return { h: r.steps, price: r.anchor, last: r.anchorTime, future: fut, bands: r.bands, colors: chartColors(), earnAt, label: r.live ? "now" : "last close" };
  }
  function paintToday() {                        // new prices for today; the days on the time axis stay the same,
    if (!state.stock || !chart) return;          // so the chart keeps whatever the person has scrolled or zoomed to
    setSeries();
    cone.set(coneData(), false);
    renderLive();
    if (analysts.has(state.sel)) renderAnalysts(analysts.get(state.sel));
  }
  function drawStock(animate, data) {
    if (data) setSeries(); else styleSeries();
    cone.set(coneData(), animate);
    setRange();
  }

  // ---------- market tape ----------
  function renderTape() {
    const m = state.index.market || [];
    $("tape").innerHTML = m.map((x, i) => (i === 4 ? '<div class="gap" aria-hidden="true"></div>' : "") +
      `<div title="${x.t}"><span class="n">${esc(x.name)}</span><span class="v">${P(x).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}<span class="${C(x) >= 0 ? "up" : "down"}">${signed(C(x))}</span></span>${spark(x.spark, 54, 26)}</div>`).join("");
    $("tape").hidden = !m.length;
  }

  // ---------- stock list ----------
  // the filters in the sidebar: sector, today's move, the model's rank and how wide the range is (both for the look-ahead chosen at the top)
  function rangeCuts() {                                   // thirds of all stocks by range width: calm, middle, wild
    const all = state.index.stocks.map((s) => s.w[state.h]).sort((a, b) => a - b);
    return [all[Math.floor(all.length / 3)], all[Math.floor((all.length * 2) / 3)]];
  }
  function keep(s, cuts) {
    const f = state.f, h = state.h;
    if (f.sec && s.sec !== f.sec) return false;
    if (f.move === "up" && !(C(s) > 0)) return false;
    if (f.move === "down" && !(C(s) < 0)) return false;
    if (f.rank === "top" && !(s.rk && s.rk[h] >= 0.8)) return false;
    if (f.rank === "bottom" && !(s.rk && s.rk[h] <= 0.2)) return false;
    if (f.range === "calm" && !(s.w[h] <= cuts[0])) return false;
    if (f.range === "wild" && !(s.w[h] >= cuts[1])) return false;
    return true;
  }
  const activeFilters = () => Object.values(state.f).filter(Boolean).length;
  function rows() {
    const h = state.h, cuts = rangeCuts();
    let r = state.index.stocks;
    if (state.filter === "star") r = r.filter((s) => state.stars.has(s.t));
    if (state.filter === "earn") r = r.filter(soon);
    r = r.filter((s) => keep(s, cuts));
    const by = { az: (a, b) => a.t.localeCompare(b.t), narrow: (a, b) => a.w[h] - b.w[h], wide: (a, b) => b.w[h] - a.w[h],
      move: (a, b) => Math.abs(C(b)) - Math.abs(C(a)),
      rank: (a, b) => (b.rk ? b.rk[h] : -1) - (a.rk ? a.rk[h] : -1), rankl: (a, b) => (a.rk ? a.rk[h] : 2) - (b.rk ? b.rk[h] : 2), earn: (a, b) => (a.next_earn || "9999").localeCompare(b.next_earn || "9999") };
    return r.slice().sort(by[state.filter === "earn" && state.sort === "az" ? "earn" : state.sort]);
  }
  function soon(s) { return !!s.next_earn && s.next_earn <= state.soonBy; }
  function scaleMax() {                                     // one width scale for every stock (the Overview's range bars)
    const all = state.index.stocks.map((s) => s.w[state.h]).sort((a, b) => a - b);
    return all[Math.floor(all.length * 0.97)] || 0.2;
  }
  // the second line of a row: the company name, or the number the list is sorted or filtered by
  function rowMeta(s) {
    const h = state.h;
    if ((state.sort === "rank" || state.sort === "rankl" || state.f.rank) && s.rk) {
      const p = s.rk[h];
      return `Ranked ${p >= 0.5 ? "top" : "bottom"} ${Math.max(1, Math.round((p >= 0.5 ? 1 - p : p) * 100))}%`;
    }
    if (state.sort === "narrow" || state.sort === "wide" || state.f.range) return `Range ±${pct(s.w[h])} in ${h} days`;
    if ((state.sort === "earn" || state.filter === "earn") && s.next_earn) return `Earnings around ${day(s.next_earn)}`;
    return s.name === s.t ? "" : esc(s.name);
  }
  function renderFilters() {
    const sel = $("f-sec"), secs = [...new Set(state.index.stocks.map((s) => s.sec).filter(Boolean))].sort();
    $("frow-sec").hidden = !secs.length;
    if (sel.options.length !== secs.length + 1)
      sel.innerHTML = `<option value="">All sectors</option>` + secs.map((x) => `<option>${esc(x)}</option>`).join("");
    sel.value = state.f.sec;
    document.querySelectorAll("[data-fk]").forEach((b) => b.setAttribute("aria-pressed", String(state.f[b.dataset.fk] === b.dataset.fv)));
    const n = activeFilters();
    $("fcount").hidden = !n; $("fcount").textContent = n;
    $("fclear").hidden = !n;
  }
  function renderList() {
    if (state.f.sec && !state.index.stocks.some((s) => s.sec === state.f.sec)) state.f.sec = "";   // a saved sector that is no longer there
    const h = state.h, list = $("list"), r = rows();
    const n = { all: state.index.stocks.length, star: state.stars.size, earn: state.index.stocks.filter(soon).length };
    document.querySelectorAll("[data-filter]").forEach((b) => {
      b.setAttribute("aria-selected", String(b.dataset.filter === state.filter));
      b.innerHTML = `${{ all: "All", star: "Starred", earn: "Earnings soon" }[b.dataset.filter]}<b>${n[b.dataset.filter]}</b>`;
    });
    renderFilters();
    $("count").textContent = r.length === state.index.stocks.length ? `${r.length} stocks` : `${r.length} of ${state.index.stocks.length} stocks`;
    $("scale").innerHTML = `Rank and range are for the next <b>${h} trading days</b>; change it with Look ahead at the top.`;
    if (!r.length) {
      list.innerHTML = activeFilters()
        ? `<li class="empty"><strong>No stock matches these filters</strong>Loosen one of them, or <button type="button" class="link" data-clear>clear all filters</button>.</li>`
        : state.filter === "star"
        ? `<li class="empty"><strong>No starred stocks yet</strong>Open a stock and press the star next to its name to keep it here.</li>`
        : `<li class="empty"><strong>Nothing to show</strong>No stock has an earnings report expected in the next 20 trading days.</li>`;
      return;
    }
    list.innerHTML = r.map((s) => `<li><button type="button" class="row" data-t="${s.t}"${s.t === state.sel ? ' aria-current="true"' : ""}>
        ${logoTag(s.t, "md")}
        <span class="id"><span class="t">${s.t}${s.earn[h] ? `<i class="e" title="${s.reported ? "Earnings were just reported" : "Earnings report expected in this window"}"></i>` : ""}${state.stars.has(s.t) ? '<span class="star-dot" title="Starred">★</span>' : ""}</span><span class="n">${rowMeta(s)}</span></span>
        ${spark(s.spark, 48, 22)}
        <span class="num"><span class="p">${money(P(s))}</span><span class="${C(s) >= 0 ? "up" : "down"}">${signed(C(s))}</span></span>
      </button></li>`).join("");
    paintLogos(list);
  }

  // ---------- selected stock ----------
  async function select(t, scroll) {
    if (!state.index.stocks.some((s) => s.t === t)) t = state.index.stocks[0].t;
    state.sel = t;
    document.querySelectorAll(".row[aria-current]").forEach((r) => r.removeAttribute("aria-current"));
    const row = document.querySelector(`.row[data-t="${CSS.escape(t)}"]`);
    if (row) {
      row.setAttribute("aria-current", "true");
      if (scroll) { const list = $("list"); list.scrollTop = row.parentElement.offsetTop - list.clientHeight / 2 + 40; }
    }
    $("chart").classList.add("loading");
    if (!cache.has(t)) cache.set(t, fetch(`data/stocks/${encodeURIComponent(t)}.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }));
    let s;
    try { s = await cache.get(t); } catch (e) {
      cache.delete(t); $("chart").classList.remove("loading");
      $("say").textContent = `${t} could not be loaded. Rebuild the data with: python -m scripts.run_dashboard --refresh`;
      return;
    }
    if (state.sel !== t) return;
    state.stock = s;
    $("chart").classList.remove("loading");
    route();
    renderStock(true, true);
    pollOne();
    loadAnalysts(t);
  }
  // the parts of the stock page that change while the market is open: price, the three ranges, the ladder
  function renderLive() {
    const s = state.stock, h = state.h, r = rangeNow(s, h), q = quote(s.t), st = state.today.state, b = r.bands["80"];
    $("price").textContent = money(q ? q.last : s.price);
    $("chg").innerHTML = q ? `${pill(q.last / q.prev - 1)}<span>today so far, ${day(q.date, true)}</span>` : `${pill(s.chg)}<span>close on ${day(s.asof, true)}</span>`;
    $("pxnote").textContent = q ? `Last close ${money(s.price)} on ${day(s.asof, true)}`
      : st ? (st.open ? "Market open: today's price is loading" : `Market closed, opens ${whenLocal(st.next_open)}`) : "";
    $("fc").innerHTML = Object.keys(s.ranges).map((hh) => { const x = rangeNow(s, hh), v = x.bands["80"];
      return `<button type="button" data-fc="${hh}" aria-pressed="${hh === h}"><span class="when"><b>${day(x.end, true)}</b><span>${x.live ? `${x.steps + 1} trading days left` : `in ${hh} trading days`}</span></span>
        <span class="rng"><span>${money(v[0])}</span><i>to</i><span>${money(v[1])}</span></span>
        <span class="how">${x.earn ? '<i class="e" title="Earnings report in this window"></i>' : ""}±${pct(x.half["80"])}, right 4 times in 5</span></button>`; }).join("");
    $("fcnote").innerHTML = r.live
      ? `<i class="livedot"></i>Following today's price: centred on ${money(q.last)} and sized for the time left to each date. At the close of ${day(s.asof, true)} the ${h}-day range was ${money(r.made.bands["80"][0])} to ${money(r.made.bands["80"][1])}.`
      : `Measured from the close of ${day(s.asof, true)}, ${money(s.price)}. While the market is open the ranges follow the price.`;
    $("say").innerHTML = `By ${day(r.end, true)}, ${s.t} is likely to be between <strong>${money(b[0])}</strong> and <strong>${money(b[1])}</strong>. ` +
      `Replayed on 2012 to 2024, ranges made at the close held the price about 4 times out of 5. They show how far the price may move, not which way.`;
    $("ladder-title").textContent = `Likely price by ${day(r.end)}`;
    const lo = r.bands["90"][0], hi = r.bands["90"][1], pos = (v) => ((v - lo) / (hi - lo)) * 100;
    $("ladder").innerHTML = ["50", "80", "90"].map((lv) => {
      const x = r.bands[lv];
      return `<div class="rung" data-l="${lv}"><span class="how">${PLAIN[lv]}</span><span class="pct">±${pct(r.half[lv])}</span>
        <span class="vals">${money(x[0])} to ${money(x[1])}</span><span></span>
        <span class="bar"><i style="left:${pos(x[0]).toFixed(1)}%;right:${(100 - pos(x[1])).toFixed(1)}%"></i><b style="left:${pos(r.anchor).toFixed(1)}%"></b></span></div>`;
    }).join("");
    $("ladder-note").textContent = `The mark is ${r.live ? "the price now" : "the last close"}, ${money(r.anchor)}. A narrower range is right less often; pick the one that matches how sure you need to be.`;
  }
  // where the trained model ranks this stock among all of them, for each look-ahead
  const ordinal = (n) => n + (n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"][n % 10]);
  function renderRank() {
    const s = state.stock, R = s.rank, M = state.track.model, h = state.h, box = $("rankbox");
    if (!R || !M) {
      box.innerHTML = `<p class="fine">The ranking model is not saved for daily use yet. Train it once with <b>python -m scripts.kaggle_cloud final-model</b>; after that every stock is ranked here after each close.</p>`;
      return;
    }
    const x = R[h], K = (state.track.ranking || {})[h];
    const group = x.pct >= 0.8 ? "top fifth" : x.pct <= 0.2 ? "bottom fifth" : "middle";
    box.innerHTML = Object.keys(R).map((hh) => `<button type="button" class="rk" data-fc="${hh}" aria-pressed="${hh === h}">
        <span>${hh} days</span><span class="line"><b style="left:${(R[hh].pct * 100).toFixed(1)}%"></b></span><span>${ordinal(R[hh].pos)} of ${M.n}</span></button>`).join("") +
      `<div class="rk-ends"><span>ranked lowest</span><span>ranked highest</span></div>
      <p class="rk-say">For the next ${h} trading days the model ranks ${s.t} <strong>${ordinal(x.pos)} of ${M.n}</strong>, in the <strong>${group}</strong>.` +
      (K ? ` In 13 years of tests, the bottom fifth beat the S&P 500 ${pct(K.bottom20_beat)} of the time and the top fifth ${pct(K.top20_beat)}: the ranking is better at flagging likely laggards than at picking winners.` : "") + `</p>
      <p class="fine">A rank is the model's order of all stocks against the market, not a price. Model trained on data to ${day(M.trained_through)} ${M.trained_through.slice(0, 4)}; ranked after the close of ${day(s.asof, true)}. How the ranking did in 13 years of tests: <button type="button" class="link" data-view="record">Track record</button>.</p>`;
  }
  function renderStock(animate, data) {
    const s = state.stock, h = state.h, r = s.ranges[h], st = s.stats;
    const inWin = r.earn && !!s.earnings.next && s.earnings.next <= s.future[Math.min(s.future.length - 1, Number(h) + 2)];
    const lg = $("logo"); lg.dataset.logo = s.t; paintLogos(lg.parentElement);
    $("name").textContent = s.name || s.t;
    $("sym").textContent = s.t;
    $("sym").hidden = !s.name || s.name === s.t;
    renderLive();
    const on = state.stars.has(s.t);
    $("star").setAttribute("aria-pressed", String(on));
    $("star").setAttribute("aria-label", on ? `Remove ${s.t} from starred` : `Star ${s.t}`);
    $("chips").innerHTML =
      (s.sector ? `<span class="chip"${s.industry ? ` title="SEC industry: ${esc(s.industry)}"` : ""}>${esc(s.sector)}</span>` : "") +
      (s.earnings.reported ? `<span class="chip${r.earn && !inWin ? " amber" : ""}">${r.earn && !inWin ? '<i class="e"></i>' : ""}Reported earnings ${day(s.earnings.reported)}</span>` : "") +
      (s.earnings.next ? `<span class="chip${inWin ? " amber" : ""}">${inWin ? '<i class="e"></i>' : ""}${s.earnings.reported ? "Next earnings" : "Earnings"} around ${day(s.earnings.next)}</span>` : "") +
      `<span class="chip">Typical day<b>±${pct(s.vol)}</b></span>` +
      (st.chg5 !== null ? `<span class="chip">5 days<b class="${st.chg5 >= 0 ? "up" : "down"}">${signed(st.chg5, 1)}</b></span>` : "") +
      (st.chg20 !== null ? `<span class="chip">1 month<b class="${st.chg20 >= 0 ? "up" : "down"}">${signed(st.chg20, 1)}</b></span>` : "") +
      (st.chg252 !== null ? `<span class="chip">1 year<b class="${st.chg252 >= 0 ? "up" : "down"}">${signed(st.chg252, 1)}</b></span>` : "");
    drawStock(animate, data);
    const w = $("warn");
    if (r.earn && !inWin && s.earnings.reported) {
      w.hidden = false;
      w.textContent = `${s.t} reported earnings on ${day(s.earnings.reported)}. Prices often keep moving for a few days after a report, so this range is still wider than usual.`;
    } else if (inWin) {
      w.hidden = false;
      w.textContent = `An earnings report is expected around ${day(s.earnings.next)}` + (h === "60" ? ", inside this window." : ", so this range is wider than usual.") +
        (s.earnings.typical ? ` On its last ${s.earnings.history.length} earnings days ${s.t} typically moved ${pct(s.earnings.typical)}.` : "");
    } else w.hidden = true;

    const at = (v, lo, hi) => Math.min(100, Math.max(0, ((v - lo) / (hi - lo || 1)) * 100)).toFixed(1);
    $("tiles").innerHTML =
      `<div class="tile"><p class="k">Range on ${day(s.asof)}</p><div class="span"><b style="left:${at(s.price, st.low, st.high)}%"></b></div><div class="ends"><span>${money(st.low)}</span><span>${money(st.high)}</span></div></div>` +
      `<div class="tile"><p class="k">Past 12 months</p><div class="span"><b style="left:${at(s.price, st.lo52, st.hi52)}%"></b></div><div class="ends"><span>${money(st.lo52)}</span><span>${money(st.hi52)}</span></div></div>` +
      `<div class="tile"><p class="k">Shares traded a day</p><p class="v">${big(st.avgvol)}</p><p class="s">20-day average</p></div>` +
      `<div class="tile"><p class="k">Below its 12-month high</p><p class="v">${pct(Math.max(0, 1 - s.price / st.hi52))}</p><p class="s">high ${money(st.hi52)}</p></div>`;

    renderRank();
    const e = s.earnings;
    $("sched").innerHTML = e.next
      ? (e.reported ? `<p class="when">Reported earnings on ${day(e.reported, true)}</p>` : "") +
        `<p class="when"><i class="e"></i>${e.reported ? "Next earnings" : "Earnings"} report around ${day(e.next, true)}</p>
         <p class="fine">Estimated from last year's date; the company confirms the exact day a few weeks ahead.</p>` +
        (e.history.length ? `<div class="moves" role="img" aria-label="Price move on each of the last ${e.history.length} earnings days">` +
          (() => { const m = Math.max(...e.history.map((x) => Math.abs(x.move)), 0.01);
            return e.history.map((x) => `<div title="${day(x.date)} ${x.date.slice(0, 4)}: ${signed(x.move, 1)}"><i class="${x.move >= 0 ? "up" : "down"}" style="height:${(Math.abs(x.move) / m * 48).toFixed(1)}%"></i></div>`).join(""); })() +
          `</div><p class="fine">Move on each of the last ${e.history.length} earnings days, oldest first. Typical size ${pct(e.typical)}.</p>` : "")
      : `<p class="fine">No earnings history found for this stock, so its ranges are not adjusted for earnings.</p>`;
    $("filings").innerHTML = s.filings.length
      ? s.filings.map((f) => `<li><time datetime="${f.date}">${day(f.date)}</time><span${f.sev >= 2 ? ' class="imp"' : ""}>${esc(cap(f.what))}</span></li>`).join("")
      : `<li class="none">No reports filed with the SEC in the last four months.</li>`;
  }

  // The latest weekday whose US session has finished (4:20 pm New York), as YYYY-MM-DD. Holidays are not known.
  function lastCloseDay() {
    try {
      const ny = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
      if (isNaN(ny)) return null;
      const d = new Date(ny.getFullYear(), ny.getMonth(), ny.getDate(), 12);
      if (ny.getHours() * 60 + ny.getMinutes() < 16 * 60 + 20) d.setDate(d.getDate() - 1);
      while (d.getDay() % 6 === 0) d.setDate(d.getDate() - 1);
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    } catch (e) { return null; }
  }

  // ---------- what analysts expect (12-month price targets, ratings), from the local server ----------
  const analysts = new Map();
  let anWidth = 0;
  new ResizeObserver(() => {                                   // the picture is drawn at pixel size, so redraw when its box changes
    const w = $("an-chart").clientWidth;
    if (!w || Math.abs(w - anWidth) < 2 || !state.stock || !analysts.has(state.sel)) return;
    renderAnalysts(analysts.get(state.sel));
  }).observe($("an-chart"));
  async function loadAnalysts(t) {
    const box = $("analyst");
    if (!analysts.has(t)) {
      box.hidden = true;
      try {
        const r = await fetch("api/analysts?t=" + encodeURIComponent(t));
        if (!r.ok) return;                                   // opened without the local server, or an older one
        analysts.set(t, await r.json());
      } catch (e) { return; }
    }
    if (state.sel === t && state.stock) renderAnalysts(analysts.get(t));
  }
  function renderAnalysts(a) {
    const s = state.stock, box = $("analyst"), T = a && a.target;
    if (!a) { box.hidden = true; return; }
    box.hidden = false;
    anWidth = $("an-chart").clientWidth;
    if (!T) {
      $("an-chart").innerHTML = ""; $("an-side").innerHTML = "";
      $("an-note").textContent = a.error ? `Analyst forecasts for ${s.t} could not be loaded (${a.error}).` : `No analyst price targets were found for ${s.t}.`;
      if (!a.ratings) return;
    }
    const q = quote(s.t), now = q ? q.last : s.price;
    const chg = (v) => `<tspan class="${v >= now ? "pos" : "neg"}">${signed(v / now - 1, 1)}</tspan>`;
    if (T) {
      // one year of closes on the left half, the next twelve months on the right half: the same time scale
      const hist = s.candles.slice(-252).map((c) => c[4]).concat(q ? [q.last] : []);
      const W = Math.max(300, $("an-chart").clientWidth), H = 250, x0 = 4, x1 = W - 158, xm = x0 + (x1 - x0) / 2, top = 16, bot = 30;   // drawn at real size: 1 unit = 1 pixel
      let lo = Math.min(...hist, T.low, now), hi = Math.max(...hist, T.high, now);
      const padY = (hi - lo) * 0.07 || 1; lo -= padY; hi += padY;
      const y = (v) => top + (hi - v) / (hi - lo) * (H - top - bot), x = (i) => x0 + i / (hist.length - 1) * (xm - x0);
      const line = hist.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      const ticks = [0.15, 0.5, 0.85].map((f) => lo + (hi - lo) * f);
      const ends = [["High", T.high], ["Average", T.mean], ["Low", T.low]].map(([n, v]) => ({ n, v, y: y(v) })).sort((p, r) => p.y - r.y);
      for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 17);      // keep the three labels apart
      const end = date(s.asof); end.setFullYear(end.getFullYear() + 1);
      const r60 = s.ranges["60"], xb = xm + 60 / 252 * (x1 - xm);
      $("an-chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${s.t}: the past year of prices and analysts' 12-month price targets. Average ${money(T.mean)}, high ${money(T.high)}, low ${money(T.low)}.">
        ${ticks.map((v) => `<line x1="${x0}" x2="${x1}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="var(--rule-soft)"/><text x="${x0}" y="${(y(v) - 4).toFixed(1)}">${money(v)}</text>`).join("")}
        <polygon points="${xm},${y(now).toFixed(1)} ${x1},${y(T.high).toFixed(1)} ${x1},${y(T.low).toFixed(1)}" fill="var(--cone-90)"/>
        <line x1="${xm}" y1="${y(now).toFixed(1)}" x2="${x1}" y2="${y(T.high).toFixed(1)}" stroke="var(--cone)" stroke-width="1.25" stroke-dasharray="4 4"/>
        <line x1="${xm}" y1="${y(now).toFixed(1)}" x2="${x1}" y2="${y(T.low).toFixed(1)}" stroke="var(--cone)" stroke-width="1.25" stroke-dasharray="4 4"/>
        <line x1="${xm}" y1="${y(now).toFixed(1)}" x2="${x1}" y2="${y(T.mean).toFixed(1)}" stroke="var(--cone)" stroke-width="2.25"/>
        <line x1="${xm}" x2="${xm}" y1="${top}" y2="${H - bot}" stroke="var(--rule)"/>
        <polyline points="${line}" fill="none" stroke="var(--ink)" stroke-width="1.6" stroke-linejoin="round"/>
        ${r60 ? `<line x1="${xb.toFixed(1)}" x2="${xb.toFixed(1)}" y1="${y(r60.bands["80"][1]).toFixed(1)}" y2="${y(r60.bands["80"][0]).toFixed(1)}" stroke="var(--cone)" stroke-width="5" stroke-linecap="round"/>` : ""}
        <circle cx="${xm}" cy="${y(now).toFixed(1)}" r="4.5" fill="var(--ink)" stroke="var(--sheet)" stroke-width="2"/>
        ${[T.high, T.mean, T.low].map((v) => `<circle cx="${x1}" cy="${y(v).toFixed(1)}" r="3.5" fill="var(--cone)"/>`).join("")}
        ${ends.map((e) => `<text x="${x1 + 10}" y="${(e.y + 4).toFixed(1)}">${e.n} <tspan class="v">${money(e.v)}</tspan> ${chg(e.v)}</text>`).join("")}
        <text x="${x0}" y="${H - 8}">${date(s.candles[Math.max(0, s.candles.length - 252)][0]).toLocaleDateString("en-GB", { month: "short", year: "numeric" })}</text>
        <text x="${xm}" y="${H - 8}" text-anchor="middle">${q ? "Now" : "Last close"} <tspan class="v">${money(now)}</tspan></text>
        <text x="${x1}" y="${H - 8}" text-anchor="end">${end.toLocaleDateString("en-GB", { month: "short", year: "numeric" })}</text>
      </svg>`;
    }
    const R = a.ratings, n = R ? R.strongBuy + R.buy + R.hold + R.sell + R.strongSell : 0, beat = (a.eps || []).filter((e) => e.actual >= e.estimate).length;
    const seg = [["sb", "Strong buy", "strongBuy"], ["b", "Buy", "buy"], ["h", "Hold", "hold"], ["s", "Sell", "sell"], ["ss", "Strong sell", "strongSell"]];
    $("an-side").innerHTML = (T ? `<div><p class="k">Average target${T.analysts ? `, ${T.analysts} analysts` : ""}</p>
      <p class="big">${money(T.mean)}${pill(T.mean / now - 1, 1)}</p>
      <dl><dt>Highest</dt><dd>${money(T.high)}</dd><dt>Lowest</dt><dd>${money(T.low)}</dd>${T.median ? `<dt>Middle (median)</dt><dd>${money(T.median)}</dd>` : ""}</dl></div>` : "") +
      (n ? `<div><h3>Ratings this month</h3>
      <div class="rate" role="img" aria-label="${seg.map(([, l, k]) => `${R[k]} ${l.toLowerCase()}`).join(", ")}">${seg.map(([c, , k]) => (R[k] ? `<i class="${c}" style="flex:${R[k]}"></i>` : "")).join("")}</div>
      <div class="rate-key">${seg.map(([c, l, k]) => `<i class="${c}"></i><span>${l}</span><b>${R[k]}</b>`).join("")}</div>
      <p class="eps">${R.strongBuy + R.buy} of ${n} analysts say buy.</p></div>` : "") +
      ((a.eps || []).length ? `<div><h3>Last ${a.eps.length} earnings reports</h3><p class="eps">Profit per share beat the analysts' estimate ${beat} time${beat === 1 ? "" : "s"} out of ${a.eps.length}.</p></div>` : "");
    if (T || n) $("an-note").textContent = `Collected from ${a.source || "Yahoo Finance"} on ${day(a.date)}. These are analysts' opinions, not LatticeIQ forecasts, and LatticeIQ has not tested how often they come true.` +
      (T && s.ranges["60"] ? ` The thick blue bar is LatticeIQ's own 60-day range, for scale.` : "");
  }

  // ---------- overview page ----------
  function nextDays(from, n) {
    const out = [], d = date(from);
    while (out.length < n) { d.setDate(d.getDate() + 1); if (d.getDay() % 6) out.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`); }
    return out;
  }
  function renderOverview() {
    const ix = state.index, S = ix.stocks, h = state.h;
    const liveQ = S.map((s) => quote(s.t)).filter(Boolean), liveN = liveQ.length, liveDay = liveN ? liveQ[0].date : null;
    const up = S.filter((s) => C(s) > 0.0005).length, dn = S.filter((s) => C(s) < -0.0005).length, flat = S.length - up - dn;
    const days = nextDays(ix.asof, 10);
    const byDay = Object.fromEntries(days.map((d) => [d, []]));
    S.forEach((s) => { if (s.next_earn && byDay[s.next_earn]) byDay[s.next_earn].push(s); });
    const due = S.filter((s) => s.next_earn && s.next_earn <= ix.asof).length, done = S.filter((s) => s.reported).length;
    const nEarn = days.reduce((a, d) => a + byDay[d].length, 0);
    const sorted = S.slice().sort((a, b) => C(b) - C(a));
    const maxMove = Math.max(Math.abs(C(sorted[0])), Math.abs(C(sorted[sorted.length - 1])), 0.001);
    const byW = S.slice().sort((a, b) => a.w[h] - b.w[h]), max = scaleMax();
    const mv = (s, kind) => `<button type="button" class="mv" data-t="${s.t}">${logoTag(s.t)}
      <span><span class="t">${s.t}</span><span class="n">${s.name === s.t ? "" : esc(s.name)}</span></span>
      ${kind === "r" ? `<span class="bar"><i class="${s.rk[h] >= 0.5 ? "up" : "down"}" style="width:${(Math.abs(s.rk[h] - 0.5) * 200).toFixed(1)}%"></i></span><span class="val">${ordinal(ranked.indexOf(s) + 1)}</span>`
        : kind === "w" ? `<span class="bar"><i class="w" style="width:${Math.min(100, s.w[h] / max * 100).toFixed(1)}%"></i></span><span class="val">±${pct(s.w[h])}</span>`
        : `<span class="bar"><i class="${C(s) >= 0 ? "up" : "down"}" style="width:${(Math.abs(C(s)) / maxMove * 100).toFixed(1)}%"></i></span>${pill(C(s))}`}</button>`;
    const ranked = S.filter((s) => s.rk).sort((a, b) => b.rk[h] - a.rk[h]);
    const flags = S.filter((s) => s.flag).sort((a, b) => b.flag.date.localeCompare(a.flag.date)).slice(0, 8);
    $("overview").innerHTML = `<div>
      <h1>${liveN ? `Markets today, ${day(liveDay, true)}` : `Markets on ${day(ix.asof, true)}`}</h1>
      <p class="lede">${liveN ? `So far today ${up} of ${S.length} stocks in the S&amp;P 500 are up and ${dn} are down.` : `${up} of ${S.length} stocks in the S&amp;P 500 rose and ${dn} fell.`} ${nEarn ? `${nEarn} earnings report${nEarn === 1 ? " is" : "s are"} expected in the next two weeks; those are the days prices move most.` : "No earnings reports are expected in the next two weeks."}</p>
      ${marketLine() ? `<p class="status"><i class="dot${state.today.state && state.today.state.open ? " on" : ""}"></i>${marketLine()}</p>` : ""}
      <div class="breadth" role="img" aria-label="${up} stocks up, ${flat} unchanged, ${dn} down"><i class="u" style="flex:${up}"></i>${flat ? `<i class="f" style="flex:${flat}"></i>` : ""}<i class="d" style="flex:${dn}"></i></div>
      <div class="breadth-key"><span><b>${up}</b> up</span>${flat ? `<span><b>${flat}</b> unchanged</span>` : ""}<span><b>${dn}</b> down</span></div>

      <h2>Earnings calendar</h2>
      <p>Expected dates, estimated from each company's date last year. A stock's range is widened when its report falls inside the window.${done || due ? ` ${[done ? `${done} ${done === 1 ? "company" : "companies"} reported in the last few days` : "", due ? `${due} ${due === 1 ? "was" : "were"} due and ${due === 1 ? "has" : "have"} not filed yet` : ""].filter(Boolean).join("; ")}.` : ""}</p>
      <div class="cal">${days.map((d, i) => { const L = byDay[d].sort((a, b) => a.t.localeCompare(b.t)); const wk = i > 0 && date(d).getDay() === 1;
        return `<div class="dayc${wk ? " week" : ""}"><div class="dh"><b>${day(d, true)}</b><span>${L.length ? L.length + (L.length === 1 ? " report" : " reports") : "none"}</span></div>
          ${L.slice(0, 6).map((s) => `<button type="button" class="tick" data-t="${s.t}">${logoTag(s.t, "sm")}${s.t}</button>`).join("")}
          ${L.length > 6 ? `<button type="button" class="more" data-more="${d}">${L.length - 6} more</button>` : ""}${L.length ? "" : '<p class="nil">No reports expected</p>'}</div>`; }).join("")}</div>

      <h2>${liveN ? "Biggest moves today" : `Biggest moves on ${day(ix.asof)}`}</h2>
      <p>${liveN ? `Price change so far today, against the close of ${day(ix.asof, true)}.` : "Price change over the last trading day."}</p>
      <div class="pair"><div class="board"><h3>Rose most</h3>${sorted.slice(0, 6).map((s) => mv(s)).join("")}</div>
        <div class="board"><h3>Fell most</h3>${sorted.slice(-6).reverse().map((s) => mv(s)).join("")}</div></div>

      <h2>Calmest and wildest, next ${h} trading days</h2>
      <p>How far each price may move, 4 times in 5. Change the look-ahead at the top to see other windows.</p>
      <div class="pair"><div class="board"><h3>Narrowest ranges</h3>${byW.slice(0, 6).map((s) => mv(s, "w")).join("")}</div>
        <div class="board"><h3>Widest ranges</h3>${byW.slice(-6).reverse().map((s) => mv(s, "w")).join("")}</div></div>

      ${ranked.length ? `<h2>The model's ranking, next ${h} trading days</h2>
      <p>Every stock ordered against the market by the trained model after the last close.${(state.track.ranking || {})[h] ? ` In 13 years of tests the lowest fifth beat the S&amp;P 500 ${pct(state.track.ranking[h].bottom20_beat)} of the time and the highest fifth ${pct(state.track.ranking[h].top20_beat)}.` : ""}</p>
      <div class="pair"><div class="board"><h3>Ranked highest</h3>${ranked.slice(0, 6).map((s) => mv(s, "r")).join("")}</div>
        <div class="board"><h3>Ranked lowest</h3>${ranked.slice(-6).reverse().map((s) => mv(s, "r")).join("")}</div></div>` : ""}

      ${flags.length ? `<h2>Filings worth reading</h2>
      <p>Reports to the SEC in the last six weeks that often move a price: restructuring, write-downs, ended agreements, auditor changes, listing notices.</p>
      <div class="board"><ul class="flags">${flags.map((s) => `<li>${logoTag(s.t, "sm")}<button type="button" class="link" data-t="${s.t}"><b>${s.t}</b></button><time datetime="${s.flag.date}">${day(s.flag.date)}</time><span>${esc(cap(s.flag.what))}</span></li>`).join("")}</ul></div>` : ""}
    </div>`;
    paintLogos($("overview"));
    $("overview").querySelectorAll("[data-more]").forEach((b) => b.addEventListener("click", () => {
      const L = byDay[b.dataset.more].slice(6);
      b.insertAdjacentHTML("beforebegin", L.map((s) => `<button type="button" class="tick" data-t="${s.t}">${logoTag(s.t, "sm")}${s.t}</button>`).join(""));
      paintLogos(b.parentElement); b.remove();
    }));
  }

  // ---------- track record page ----------
  function renderRecord() {
    const t = state.track, h = state.h, R = t.ranges, K = t.ranking || {};
    const row = (hh) => { const L = R[hh].levels; return `<tr><td>${hh} trading days<small>${R[hh].forecasts.toLocaleString("en-US")} forecasts</small></td>
      ${["50", "80", "90"].map((lv) => `<td>${pct(L[lv].hit_rate)}<small>aimed for ${lv}%</small></td>`).join("")}
      <td>${pct(L["80"].hit_rate_earnings)}<small>${pct(L["80"].hit_rate_other)} in other weeks</small></td>
      <td>±${pct(L["80"].avg_width / 2)}</td></tr>`; };
    const yrs = R[h].by_year;
    const lo = 0.7, hi = 0.9, at = (v) => (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * 100;
    const rank = Object.keys(K).length ? `<h2>The ranking model</h2>
      <p>A separate model ranks every stock against the others. Its edge is small but steady. It is better at spotting likely laggards than at picking winners.</p>
      <div class="scroll"><table><thead><tr><th>Look ahead</th><th>Direction right<br>against the market</th><th>Years with a<br>positive score</th><th>Top fifth beat<br>the S&amp;P 500</th><th>Bottom fifth beat<br>the S&amp;P 500</th></tr></thead><tbody>
      ${Object.keys(K).map((hh) => `<tr><td>${hh} trading days<small>${K[hh].forecasts.toLocaleString("en-US")} forecasts</small></td><td>${pct(K[hh].hit_rate)}<small>a coin flip is 50%</small></td>
        <td>${K[hh].positive_years} of ${K[hh].years}</td><td>${pct(K[hh].top20_beat)}</td><td>${pct(K[hh].bottom20_beat)}</td></tr>`).join("")}
      </tbody></table></div>` : "";
    $("record").innerHTML = `<div>
      <h1>How often were the ranges right?</h1>
      <p class="lede">Every range on this site was replayed on 2012 to 2024: each year was forecast using only what was known before it, then compared with what the price actually did.</p>
      <h2>Said and done</h2>
      <p>If a range says “4 times in 5”, the price should land inside it about 80% of the time. This is what happened.</p>
      <div class="scroll"><table><thead><tr><th>Look ahead</th><th>“Half the time”</th><th>“4 times in 5”</th><th>“9 times in 10”</th><th>“4 times in 5” in<br>earnings weeks</th><th>Typical width<br>(4 times in 5)</th></tr></thead>
      <tbody>${Object.keys(R).map(row).join("")}</tbody></table></div>
      <h2>Year by year, ${h} trading days</h2>
      <p>Share of “4 times in 5” ranges that held the price. Years far below the line are the ones to distrust: markets moved more than their recent past suggested.</p>
      <div class="years" role="img" aria-label="Hit rate of the 4-times-in-5 range in each year, against the 80% aim">
        <div class="plot">
          <div class="goal" style="bottom:${at(0.8).toFixed(1)}%"><span>aim 80%</span></div>
          ${yrs.map((y) => { const v = at(y.hit80), g = at(0.8), low = y.hit80 < 0.75;
            return `<div class="col${low ? " low" : ""}"><i style="bottom:${Math.min(v, g).toFixed(1)}%;height:${Math.abs(v - g).toFixed(1)}%"></i><b style="bottom:${v.toFixed(1)}%"></b><em style="bottom:${v.toFixed(1)}%" class="${y.hit80 >= 0.8 ? "over" : "under"}">${(y.hit80 * 100).toFixed(0)}%</em></div>`; }).join("")}
        </div>
        <div class="axis">${yrs.map((y) => `<span>${y.year}</span>`).join("")}</div>
      </div>
      ${rank}
      <h2>What this site can and cannot tell you</h2>
      <div class="claims"><div class="yes"><h3>It can</h3><ul>
        <li>Show a price range that has been right about as often as it says.</li>
        <li>Warn when an earnings report falls inside the window, and widen the range for it.</li>
        <li>Show which stocks move little and which move a lot, on one scale.</li></ul></div>
        <div class="no"><h3>It cannot</h3><ul>
        <li>Tell you whether a stock will go up or down.</li>
        <li>Promise the price stays inside a range. One time in five it will not.</li>
        <li>See surprises: sudden news can move a price further than any range.</li></ul></div></div>
      <h2>How the ranges are made</h2>
      <p>Each range is centred on today's price. Its width follows how much that stock has been moving, measured four ways and blended, and it is wider when an earnings report is expected. The widths were tuned on windows that ended by ${day(t.calibrated_to)} ${t.calibrated_to.slice(0, 4)}; nothing after that date has been used to tune or score them.</p>
    </div>`;
  }

  // ---------- search ----------
  function hits(q) {
    q = q.trim().toLowerCase();
    if (!q) return [];
    const S = state.index.stocks;
    const a = S.filter((s) => s.t.toLowerCase() === q), b = S.filter((s) => s.t.toLowerCase().startsWith(q) && s.t.toLowerCase() !== q);
    const c = S.filter((s) => !s.t.toLowerCase().startsWith(q) && s.name.toLowerCase().includes(q));
    return a.concat(b, c).slice(0, 8);
  }
  function renderHits() {
    const q = $("q").value, box = $("hits"), H = hits(q);
    state.hits = H; state.hit = 0;
    box.hidden = !q.trim();
    $("q").setAttribute("aria-expanded", String(!box.hidden));
    box.innerHTML = H.length ? H.map((s, i) => `<li role="option" data-t="${s.t}" aria-selected="${i === 0}">${logoTag(s.t)}<span><span class="t">${s.t}</span><span class="n">${s.name === s.t ? "" : esc(s.name)}</span></span><span>${money(P(s))} ${pill(C(s))}</span></li>`).join("")
      : `<li class="none">No stock matches “${esc(q.trim())}”. Try a ticker such as NVDA, or part of a company name.</li>`;
    paintLogos(box);
  }
  function closeHits() { $("hits").hidden = true; $("q").setAttribute("aria-expanded", "false"); }
  function open(t) { closeHits(); $("q").value = ""; $("q").blur(); setView("stocks"); select(t, true); }

  // ---------- keeping today's prices fresh ----------
  function refreshLive() {
    renderTape();
    renderList();
    if (state.view === "overview") renderOverview();
    paintToday();
    footer();
  }
  async function pollToday() {
    try {
      const r = await fetch("api/today");
      if (!r.ok) return;
      const j = await r.json();
      if (state.built && j.built && j.built !== state.built) { location.reload(); return; }   // forecasts rolled to a new day
      state.built = state.built || j.built;
      const was = state.today, changed = j.fetched !== was.fetched || !was.state || j.state.open !== was.state.open || j.state.next_open !== was.state.next_open;
      const mine = state.sel && was.quotes[state.sel];
      state.today = { quotes: j.quotes || {}, state: j.state, fetched: j.fetched };
      if (mine && (!j.quotes[state.sel] || j.quotes[state.sel].date === mine.date)) state.today.quotes[state.sel] = mine;   // keep the fresher one
      if (changed) refreshLive();
    } catch (e) { /* opened without the local server: closing prices only */ }
  }
  async function pollOne() {
    const st = state.today.state, t = state.sel;
    if (!st || !st.open || !t || state.view !== "stocks" || document.hidden) return;
    try {
      const r = await fetch("api/quote?t=" + encodeURIComponent(t));
      if (!r.ok) return;
      const j = await r.json();
      if (j.quote && j.quote.last && state.sel === t) {
        state.today.quotes[t] = j.quote;
        if (!state.today.fetched) state.today.fetched = new Date().toISOString();
        paintToday();
      }
    } catch (e) { /* keep what is shown */ }
  }
  function footer() {
    const ix = state.index, built = ix.built ? new Date(ix.built) : null;
    $("asof").textContent = `Closing prices of ${day(ix.asof, true)} ${ix.asof.slice(0, 4)} for ${ix.stocks.length} stocks in the S&P 500.` +
      (built ? ` Checked ${built.toLocaleDateString("en-GB", { day: "numeric", month: "short" })}, ${clock(ix.built)}.` : "");
  }

  // ---------- wiring ----------
  function route() { history.replaceState(null, "", state.view === "stocks" && state.sel ? `#${state.sel}/${state.h}` : `#${state.view}`); }
  function setView(v) {
    state.view = v;
    for (const id of ["overview", "stocks", "record"]) $(id).hidden = id !== v;
    document.querySelectorAll(".views [data-view]").forEach((b) => b.toggleAttribute("aria-current", false));
    document.querySelector(`.views [data-view="${v}"]`).setAttribute("aria-current", "page");
    if (v === "overview") renderOverview();
    if (v === "record") renderRecord();
    if (v === "stocks") { if (!state.sel) select((state.index.stocks.find((s) => s.t === "NVDA") || state.index.stocks[0]).t, true); else if (state.stock) drawStock(false, false); }
    route();
    window.scrollTo(0, 0);
  }
  function setH(h) {
    state.h = h;
    document.querySelectorAll("[data-h]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.h === h)));
    renderList();
    if (state.stock) renderStock(true, false);
    if (state.view === "record") renderRecord();
    if (state.view === "overview") renderOverview();
    route();
  }
  function setTheme(next) {
    if (next) { document.documentElement.dataset.theme = next; store.set("lattice-theme", next); }
    themeChart();
  }
  function step(by) {
    const r = rows(), i = r.findIndex((s) => s.t === state.sel);
    if (r.length) select(r[Math.min(r.length - 1, Math.max(0, i + by))].t, true);
  }

  async function start() {
    const saved = store.get("lattice-theme"); if (saved) document.documentElement.dataset.theme = saved;
    if (store.get("lattice-chart") === "candles") state.ctype = "candles";
    try { state.stars = new Set(JSON.parse(store.get("lattice-stars") || "[]")); } catch (e) { state.stars = new Set(); }
    let index, track;
    try {
      [index, track] = await Promise.all(["data/index.json", "data/track.json"].map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); })));
    } catch (e) {
      $("overview").hidden = false;
      $("overview").innerHTML = `<div><h1>No data yet</h1><p class="lede">In the project folder run <strong>python -m scripts.run_dashboard</strong>. It builds the data from your price files and opens this page.</p></div>`;
      return;
    }
    state.index = index; state.track = track;
    state.soonBy = nextDays(index.asof, 20)[19];
    const built = index.built ? new Date(index.built) : null;
    footer();
    const want = lastCloseDay(), old = built ? Date.now() - built.getTime() > 3 * 3600e3 : true;
    if (index.source !== "live" || (want && index.asof < want && old)) {
      $("stale").hidden = false;
      $("stale").innerHTML = (index.source !== "live"
        ? `These are saved prices from <b>${day(index.asof, true)}</b>, not the latest ones.`
        : `These prices are from the close of <b>${day(index.asof, true)}</b>. Newer ones should be available.`) +
        ` To update, stop the dashboard (Ctrl+C in its window) and start it again: <code>python -m scripts.run_dashboard</code>`;
    }
    $("credit-logodev").hidden = !cfg.logoDevToken;
    $("credit-tickerlogos").hidden = !!cfg.logoDevToken || cfg.useTickerLogos === false;
    renderTape();
    buildChart();
    renderList();
    const [a, b] = decodeURIComponent(location.hash.slice(1)).split("/");
    if (["5", "20", "60"].includes(b)) { state.h = b; document.querySelectorAll("[data-h]").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.h === b))); renderList(); }
    if (a === "record" || a === "overview" || !a) setView(a || "overview");
    else if (a === "stocks") setView("stocks");
    else { state.sel = null; setView("stocks"); select(a, true); }

    document.addEventListener("click", (e) => {
      const f = e.target.closest("[data-fc]");
      if (f) { setH(f.dataset.fc); return; }
      const v = e.target.closest("[data-view]");
      if (v) { e.preventDefault(); setView(v.dataset.view); return; }
      const t = e.target.closest("[data-t]");
      if (t && !t.closest("#hits")) {
        const wasStocks = state.view === "stocks";
        if (!wasStocks) setView("stocks");
        select(t.dataset.t, !wasStocks || !t.classList.contains("row"));
        if (window.innerWidth <= 780 || window.scrollY > 120) window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
      }
      if (!e.target.closest(".search")) closeHits();
    });
    window.addEventListener("hashchange", () => {                     // a pasted or edited address, back and forward
      const [x, y] = decodeURIComponent(location.hash.slice(1)).split("/");
      if (["5", "20", "60"].includes(y) && y !== state.h) setH(y);
      if (["overview", "stocks", "record"].includes(x)) { if (x !== state.view) setView(x); }
      else if (x && x !== state.sel) { if (state.view !== "stocks") setView("stocks"); select(x, true); }
    });
    $("q").addEventListener("input", renderHits);
    $("q").addEventListener("focus", () => { if ($("q").value.trim()) renderHits(); });
    $("q").addEventListener("keydown", (e) => {
      const H = state.hits || [];
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!H.length) return;
        state.hit = (state.hit + (e.key === "ArrowDown" ? 1 : -1) + H.length) % H.length;
        $("hits").querySelectorAll("li").forEach((li, i) => li.setAttribute("aria-selected", String(i === state.hit)));
      } else if (e.key === "Enter" && H.length) open(H[state.hit].t);
      else if (e.key === "Escape") { closeHits(); $("q").blur(); }
    });
    $("hits").addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-t]"); if (li) { e.preventDefault(); open(li.dataset.t); } });
    document.addEventListener("keydown", (e) => {
      const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
      if (e.key === "/" && !typing) { e.preventDefault(); $("q").focus(); }
      else if (!typing && state.view === "stocks" && (e.key === "ArrowDown" || e.key === "ArrowUp")) { e.preventDefault(); step(e.key === "ArrowDown" ? 1 : -1); }
    });
    $("sort").addEventListener("change", (e) => { state.sort = e.target.value; renderList(); });
    try { Object.assign(state.f, JSON.parse(store.get("lattice-filters") || "{}")); } catch (e) { /* none saved */ }
    const setFilter = (k, v) => { state.f[k] = v; store.set("lattice-filters", JSON.stringify(state.f)); renderList(); $("list").scrollTop = 0; };
    $("f-sec").addEventListener("change", (e) => setFilter("sec", e.target.value));
    document.querySelectorAll("[data-fk]").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.fk, b.dataset.fv)));
    const clearFilters = () => { state.f = { sec: "", move: "", rank: "", range: "" }; store.set("lattice-filters", "{}"); renderList(); };
    $("fclear").addEventListener("click", clearFilters);
    $("list").addEventListener("click", (e) => { if (e.target.closest("[data-clear]")) clearFilters(); });
    const setOpen = (on) => {
      $("ftoggle").setAttribute("aria-expanded", String(on)); $("fbody").hidden = !on;
      store.set("lattice-filters-open", on ? "1" : "0");
    };
    $("ftoggle").addEventListener("click", () => setOpen($("ftoggle").getAttribute("aria-expanded") !== "true"));
    setOpen(store.get("lattice-filters-open") !== "0" && window.innerWidth > 780);
    renderList();
    document.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => { state.filter = b.dataset.filter; renderList(); }));
    document.querySelectorAll("[data-h]").forEach((b) => b.addEventListener("click", () => setH(b.dataset.h)));
    document.querySelectorAll("[data-p]").forEach((b) => b.addEventListener("click", () => {
      state.period = b.dataset.p;
      document.querySelectorAll("[data-p]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      if (state.stock) drawStock(false, false);
    }));
    const setType = (c) => {
      state.ctype = c === "candles" ? "candles" : "line";
      document.querySelectorAll("[data-c]").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.c === state.ctype)));
      store.set("lattice-chart", state.ctype);
      styleSeries();
    };
    document.querySelectorAll("[data-c]").forEach((b) => b.addEventListener("click", () => setType(b.dataset.c)));
    setType(store.get("lattice-chart"));
    $("chart").addEventListener("mouseleave", () => { tip.hidden = true; });
    const setVol = (on) => {
      state.vol = on; store.set("lattice-vol", on ? "1" : "0");
      $("volbtn").setAttribute("aria-pressed", String(on));
      styleSeries();
    };
    $("volbtn").addEventListener("click", () => setVol(!state.vol));
    setVol(store.get("lattice-vol") !== "0");
    // Larger chart: the card is moved to the end of the page (its normal place cannot hold a full-window box)
    // and put back where it was on closing.
    const card = $("chartcard"), home = document.createComment("chart");
    const setMax = (on) => {
      if (on === card.classList.contains("max")) return;
      if (on) { card.before(home); document.body.append(card); } else home.replaceWith(card);
      card.classList.toggle("max", on);
      $("maxbtn").setAttribute("aria-pressed", String(on));
      $("maxbtn").setAttribute("aria-label", on ? "Back to the normal chart size" : "Make the chart larger");
      requestAnimationFrame(() => setRange());
    };
    $("maxbtn").addEventListener("click", () => setMax(!card.classList.contains("max")));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMax(false); });
    document.addEventListener("click", (e) => { if (e.target.closest("[data-view], [data-t]")) setMax(false); }, true);
    $("star").addEventListener("click", () => {
      const t = state.sel; if (!t) return;
      state.stars.has(t) ? state.stars.delete(t) : state.stars.add(t);
      store.set("lattice-stars", JSON.stringify([...state.stars]));
      renderList(); renderStock(false, false);
    });
    $("theme").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
      setTheme(dark ? "light" : "dark");
    });
    $("pick").addEventListener("click", () => document.querySelector(".rail").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" }));
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => setTheme());
    pollToday();
    setInterval(pollToday, 30000);
    setInterval(pollOne, 20000);
  }
  start();
})();
