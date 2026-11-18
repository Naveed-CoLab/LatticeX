# Put the LatticeIQ dashboard on AWS

One small Linux server runs the same program as on the PC (`python -m scripts.run_dashboard`): it downloads each finished trading day, ranks the stocks, rebuilds the ranges and serves the site. A web server in front of it (Caddy) adds HTTPS, and every page asks for a password.

```
browser ──HTTPS──> Caddy (port 443, free certificate) ──> run_dashboard (127.0.0.1:8600, login) ──> data/ on the server
```

**Private on purpose.** The price data (Yahoo) and FinBERT are licensed for research, not public use, so the site needs a login until licensed feeds and a legal review are in place.

## What it costs

| Choice | Monthly | Notes |
|---|---|---|
| **Lightsail 1 GB (recommended for the $10 budget)** | **$7** | 2 vCPUs, 40 GB disk, 2 TB transfer, public IP included. The daily ranking briefly needs 1.8 GB, so it uses swap for about a minute after each close |
| Lightsail 2 GB | $12 | No swapping; pages stay fast while the daily update runs |

Prices from the Lightsail bundle list (checked 17 Nov 2026). On a new account's free plan the credits ($100, up to $200) cover this; the free plan ends after 6 months or when the credits run out, and AWS then **suspends the account** unless it is upgraded to the paid plan (Settings › Billing › Upgrade). Measured on our data: building the site 0.7 GB peak, daily ranking 1.7–1.8 GB peak.

## Steps

### 1. Create the server (AWS console, about 5 minutes)

1. Open **Lightsail** in the console, Region **Sydney (ap-southeast-2)**. If your account type does not offer Lightsail, use EC2 with Ubuntu 24.04 and a 1–2 GB instance instead; the rest is the same.
2. **Create instance** › Linux/Unix › **OS only › Ubuntu 24.04 LTS** › plan **$7 (1 GB)** › name `latticeiq` › Create.
3. **Networking** tab: **Create static IP** and attach it to `latticeiq` (free while attached). Note the address, e.g. `13.54.0.1`.
4. Same tab, **IPv4 firewall**: add a rule **HTTPS (TCP 443)**. Keep SSH (22) and HTTP (80, needed once for the certificate).
5. **Account › SSH keys**: download the default key for Sydney (`LightsailDefaultKey-ap-southeast-2.pem`) to `Downloads`.

### 2. Install LatticeIQ on the server

In Lightsail click **Connect using SSH** (a terminal opens in the browser), then:

```bash
git clone https://github.com/Naveed-CoLab/LatticeX.git
cd LatticeX
sudo bash deploy/aws/setup.sh
```

About 5 minutes. It ends by printing your site's address, `https://13-54-0-1.sslip.io` for the IP above. ([sslip.io](https://sslip.io) is a free name that points at any IP written with dashes, so HTTPS works without buying a domain. With your own domain: `sudo DOMAIN=app.example.com bash deploy/aws/setup.sh`.)

### 3. Copy the data from the PC

First run the dashboard on the PC once (`python -m scripts.run_dashboard`), so the price-range fit is saved in `data\live\range_fit.json` and the server does not have to redo it. Then, in PowerShell in `D:\Lattice IQ`:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\aws\upload_data.ps1 -Ip 13.54.0.1 -Key "$HOME\Downloads\LightsailDefaultKey-ap-southeast-2.pem"
```

It copies the frozen prices, fundamentals, SEC filings table, the trained model and the saved range fit (about 150 MB).

### 4. Set the login and start

On the server:

```bash
nano ~/LatticeX/.env
```

Fill in the two lines (Ctrl+O, Enter, Ctrl+X to save):

```
SEC_USER_AGENT=Your Name your.email@example.com
LATTICEIQ_PASSWORD=a long password only your team knows
```

Then:

```bash
sudo systemctl start latticeiq
journalctl -u latticeiq -f        # watch it start (Ctrl+C stops watching, not the site)
```

The first start downloads three years of prices and checks the SEC filings (5–10 minutes) before the site answers; until then the address shows an error. Open the address from step 2: any user name, your password.

## Day to day

| Task | Command (on the server) |
|---|---|
| Is it running? | `systemctl status latticeiq` |
| What is it doing? | `journalctl -u latticeiq -n 50` |
| Update the code after a push | `cd ~/LatticeX && git pull && sudo systemctl restart latticeiq` |
| New trained model | copy `models\latticeiq_v1.npz` again with `upload_data.ps1`, then restart |
| Change the password | edit `.env`, then `sudo systemctl restart latticeiq` |
| Stop paying | Lightsail › delete the instance **and** release the static IP |

After each US close the server downloads the finished day, ranks the stocks and rebuilds the pages by itself, as on the PC.

## If something goes wrong

| Log shows | Cause | Fix |
|---|---|---|
| `Fitting the ranges ...` then `oom-kill` | The server did not reuse the range fit and tried to redo it (needs about 4.5 GB) | `sudo systemctl stop latticeiq`; on the PC run `python -m scripts.run_dashboard` once, then `upload_data.ps1`, then start again. Since 18 Nov the fit no longer depends on line endings or file times, and a small server stops with this advice instead of crashing |
| `ssh_dispatch_run_fatal ... Connection timed out` at the start of `upload_data.ps1` | A brief network drop on the first connection | Harmless if the file lines follow without errors; otherwise run it again |
| `Not available from Yahoo (6): ANSS CA FL ...` | Former members that no longer trade | Nothing; they are used only for comparison |

## Known limits

- **Yahoo from a data centre.** Daily prices come from Yahoo, which sometimes limits requests from cloud servers. If the log shows price downloads failing, the site keeps showing the last good day; the fix is a licensed price feed, which a commercial launch needs anyway.
- **Basic login.** One shared password, sent only over HTTPS. Per-person logins (e.g. Cloudflare Access, free for 50 users, needs a domain) are the next step before showing it outside the team.
- **One server.** No backup copy runs elsewhere. Everything on it can be rebuilt from the repository plus the files in step 3.
