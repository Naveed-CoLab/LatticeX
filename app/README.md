# Dashboard

A local site for the S&P 500: a market overview, a page for any stock, and a track record.

```bash
python -m scripts.run_dashboard               # update if a newer trading day exists, then open http://localhost:8600
python -m scripts.run_dashboard --no-update   # open with the data on disk (no internet needed)
python -m scripts.run_dashboard --refresh     # rebuild the page data even if nothing changed
```

## Keeping it current

Start the dashboard once a day and it is current. Each start:

1. checks whether a newer closing day should exist (US markets close at 4 pm New York time, which is 2 am in Pakistan) and, if so, downloads the latest closing prices for every current member and the market funds (about a minute);
2. once a day, asks the SEC for filings made since the last check (a few minutes), so reported earnings and important filings are up to date;
3. rebuilds the page data (about 10 seconds).

## Today's prices and live ranges

While the dashboard is running and the US market is open (9:30 am to 4 pm New York, which is 7:30 pm to 2 am in Pakistan in winter), it also shows today's prices so far:

- the stock you have open: refreshed every 20 seconds;
- every stock, the market strip and the Overview page: refreshed every 10 minutes.

The big price becomes today's price, with the last close underneath. Before the market opens the page says when it opens in your local time and shows the last close, because no newer price exists yet. Prices come from Yahoo and can be delayed by up to 15 minutes; quotes to the second need a paid or broker feed.

**The ranges follow the price.** Each range is made at the last close. While the market is open the page moves it to the price now and sizes it for the trading time left to the same end date (each side's distance from the price × √(days left ÷ days in the window)), so the three ranges, the target day's middle, high and low, the cone on the chart and the ladder update with every new price. The line under the three ranges shows what the range was at the close. The track record was measured on ranges made at the close; the live version is the same range re-centred, not a separately tested model.

About 20 minutes after the close the dashboard downloads the finished day, rebuilds every forecast from it and the page reloads itself, so it can be left running. `start_dashboard.bat` in the project folder starts it in a minimised window with a double-click.

## Finding stocks: the sidebar

The list on the left of the Stocks page can be narrowed down and sorted:

- **All / Starred / Earnings soon** at the top.
- **Filters** (the panel opens and closes; the number shows how many are on; "Clear filters" resets them):
  - **Sector**: from the industry code each company reports to the SEC (for example 3674 semiconductors → Technology), sorted into 11 sectors. The codes are collected by the daily SEC filings check (`data/live/companies.json`); until the first check after this was added, the sector filter is not shown. It is an approximation of the usual sector names, which belong to a commercial classification.
  - **Today**: stocks up or down so far today.
  - **Rank**: the model's top or bottom 20% against the market.
  - **Range**: calm (the narrowest third of ranges) or wild (the widest third).
- **Sort** next to the count of stocks shown. When the list is sorted or filtered by rank, range or earnings, the second line of each row shows that number instead of the company name.

Rank and range filters follow the look-ahead chosen at the top (5, 20 or 60 days). The filters are remembered in this browser.

On each stock page, **Against the market** (the model's rank for 5, 20 and 60 days and what that rank did in testing) sits right under the price, before the price ranges.

## The chart

- **Line or candles.** A line with a soft fill by default (green when the chosen period is up, red when it is down) or candles; the choice is remembered.
- **Period.** 1 week to 1 year of history. The line above the chart gives the change over that period and its highest and lowest close; the same high and low are marked on the chart.
- **Earnings.** An "E" marks each past earnings report; a dashed amber line marks the next expected one when it falls inside the look-ahead.
- **The cone.** The likely range ahead, right about 4 times in 5 (the 80% range). One range is shown by default so the page is easy to read; "Show 50% and 90% ranges" under the chart or in the side panel adds the narrower range (right half the time) and the wider one (right 9 times in 10). The choice is remembered in the browser, and the API and the Track record page always keep all three.
- **Uneven sides.** Each side of a range is fitted on its own, so the range sits a little above today's price (prices have tended to drift up); the cards and the side panel show it as down / up, e.g. −8.4% / +9.8%. A dotted line in the cone is the likely middle.
- **Pointing at it.** A floating label shows the price and day's change for a past day, today so far, or the likely range on a future day; the price and date also appear on the axes.
- **On the day itself** (side panel, under the ranges): for the last day of the chosen look-ahead, the likely middle for the close, a high the day should stay below and a low it should stay above, each right about 9 times in 10, with how often that held on 2012–2024. The Track record page has the same table for all three look-aheads.
- **Buttons.** The bars button shows or hides volume (remembered). The corner button opens the chart across the whole window; Esc or the same button closes it.
- While the market is open the end of the line pulses and a dotted line marks the price now.

## Rank against the market

Once the model is trained for daily use (`python -m scripts.kaggle_cloud final-model`, see the main README), the dashboard ranks every stock with it after each close. Each stock page shows where the stock stands among all of them for 5, 20 and 60 trading days, with what that part of the ranking did in 13 years of tests. The stock list can be sorted by the model's rank, and the Overview lists the highest and lowest ranked. A rank is an order against the market, not a price, and the page says the ranking is better at flagging likely laggards than at picking winners. Until a model is saved, the panel says how to train it.

## What analysts expect

Each stock page has a card in the style of the price-target pages on the big finance sites: the past year of prices, then lines fanning out to the highest, average and lowest 12-month price target, with the buy / hold / sell counts and how often recent earnings beat the estimates. LatticeIQ's own 60-day range is drawn on the same picture as a thick blue bar, for scale.

- The numbers come from Yahoo Finance and are fetched when you open a stock, once a day per stock (`src/ingest/analysts.py`, saved under `data/live/analysts/`). Check one from the terminal with `python -m src.ingest.analysts NVDA`.
- **They are analysts' opinions, not LatticeIQ forecasts.** LatticeIQ has not tested how often they come true, and the page says so. Yahoo only gives today's consensus, so each day's numbers are kept in a small history; after a few months that history can be used to test whether targets or rating changes say anything about returns.
- If Yahoo has nothing for a stock, or the request fails, the card says so; the rest of the page does not depend on it.

## The LatticeIQ logo

The mark is three candlesticks (the past: hollow = closed up, filled = closed down) running into the forecast cone (the likely range ahead, widening with the square root of time in the chart's three band strengths). It is the picture the dashboard draws for every stock, reduced to a few shapes. The wordmark is Archivo: "Lattice" wide and heavy, "IQ" narrow and light.

| File | Use |
|---|---|
| `app/favicon.svg` | Browser-tab icon (two candles on a blue tile, readable at 16 px) |
| `docs/assets/logo.svg`, `logo_dark.svg` | Mark and name, for light and dark backgrounds (slides, report, README); the letters are outlines, so no font is needed |
| `docs/assets/logo_mark.svg`, `logo_mark_dark.svg` | The mark alone |

In the page header the mark is drawn inline in the page's own colours, so it follows the light and dark themes.

## Company logos

Logos load in your browser from an outside service. Every logo is shown as the same rounded square: a logo that comes as a solid square fills it edge to edge, a logo with a see-through background sits on a light tile, and a stock without a logo shows its first letters on a tint.

| Setting in `config.js` | Result |
|---|---|
| Nothing changed | Logos from `cdn.tickerlogos.com` (no sign-up) for the companies listed in `logos.js`: 492 of the 500 stocks on 16 Nov 2026, each address checked against the service. Images are 128-pixel squares. The footer shows the "Logos by AllInvestView" link their terms ask for |
| `logoDevToken: "pk_..."` | All stocks, from [logo.dev](https://www.logo.dev) (free publishable key), at double resolution with see-through backgrounds and separate light and dark versions. The footer shows "Logos provided by Logo.dev" |
| `useTickerLogos: false` and no key | Letter tiles only; nothing is requested from outside |

The logos are not downloaded into the project or changed: the tickerlogos terms do not allow bulk download or editing, and logos are trademarks of their owners. Some companies' logos have a white background of their own; in dark mode those show as a white square. Check both services' terms again before a commercial launch.

**Company names** come from the SEC's list and are tidied for display ("Church & Dwight Co Inc /De/" becomes "Church & Dwight"); see `tidy_name` in `scripts/export_dashboard.py`.

## Not built yet

A live record that scores the logged daily forecasts against what happened, and a chat assistant. Crash risk, recovery odds and news sentiment appear only after they pass their tests (see the main README).
