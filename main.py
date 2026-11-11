"""Roblox Up-and-Coming top-10 forecaster for make.fun.

Reads /input/history.ndjson and /input/task.json and writes /output/forecast.json: for each of the
chart's top 10 places, a probability for each game named there, plus a top 10.

The model, per game at rank r and hour h of the scored window:
    P(place k) = m_b(h) * ( w_r(h) * [k = r] + (1 - w_r(h)) * K(k; r, sigma(h)) )
  m_b  P(still in the top 10 at h | rank band b): decays from 1 to the stationary rate 10/n for the
       top bands, rises from 0 to it for deep bands (a deep game needs time to climb; by the end of
       the window every game is equally likely to be anywhere)
  w_r  P(exactly the same place at h | place r), strongest at #1
  K    truncated discrete Gaussian around r
Everything is fitted from the history by weighted least squares; the forecast is the hour marginal
averaged over the window (each hour is scored on its own, so that mean is the proper forecast).

The model's curves are fitted against the actual read spacing in hours. They do not extrapolate observations at one read spacing to another; k=1 leaves the fitted hourly clock unchanged.

Python 3.13, standard library only, no network, no randomness: two runs write the same bytes.
"""
import json
import math
import os
import sys; sys.setrecursionlimit(10000)
from collections import defaultdict
from datetime import datetime

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
PLACES = 10
FLOOR = 1e-4
NB = 20
TAU_CAP = 25.0
WIDTH = 1.8      # kernel width multiplier: the fitted width is narrow for a 72 h mean
RESID = 0.10     # share of every place left to the games the forecast does not name
MAX_ROWS = 30    # games named per place
GAP_BINS = (2, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 27, 33, 40, 50, 62, 76, 96, 130, 200)


def band_of(r):
    if r <= 10:
        return r
    if r <= 12:
        return 11
    if r <= 15:
        return 12
    if r <= 20:
        return 13
    if r <= 25:
        return 14
    if r <= 30:
        return 15
    if r <= 40:
        return 16
    if r <= 60:
        return 17
    if r <= 80:
        return 18
    return 19


def ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def chart_of(row):
    return [x['universe_id'] for x in sorted(row['up_and_coming'], key=lambda y: y['rank'])]


def rankmap(row):
    return {u: i + 1 for i, u in enumerate(chart_of(row))}


def bin_of(g):
    for k, b in enumerate(GAP_BINS):
        if g <= b:
            return k
    return len(GAP_BINS) - 1


def gauss_kernel(r, s, n=PLACES):
    d = [abs(k - r) for k in range(1, n + 1)]
    base = min(d)
    w = [math.exp(-((x * x - base * base)) / (2 * s * s)) for x in d]
    z = sum(w)
    return [x / z for x in w]


def pairs(reads):
    t = [ts(r['at']) for r in reads]
    return [(i, j, (t[j] - t[i]) / 3600.0) for i in range(len(reads)) for j in range(i + 1, len(reads))]


def collect(reads):
    R = [rankmap(r) for r in reads]
    cells = defaultdict(lambda: defaultdict(lambda: [0] * (PLACES + 2)))
    gsum = defaultdict(lambda: [0.0, 0])
    for i, j, g in pairs(reads):
        b = bin_of(g)
        gsum[b][0] += g
        gsum[b][1] += 1
        for u, rk in R[i].items():
            if rk > 40:
                continue
            cells[rk][b][min(R[j].get(u, 11), 11)] += 1
    greps = [gsum[b][0] / gsum[b][1] if gsum[b][1] else GAP_BINS[b] for b in range(len(GAP_BINS))]
    return cells, greps


def fit(reads):
    cells, greps = collect(reads)
    nb = len(GAP_BINS)
    sig = [1.0] * nb
    for b in range(nb):
        num = den = 0.0
        for r in range(2, 41):
            row = cells.get(r, {}).get(b)
            if not row:
                continue
            for k in range(1, 11):
                if k == r:
                    continue
                c = row[k]
                if c:
                    num += c * (k - r) ** 2
                    den += c
        sig[b] = math.sqrt(num / den) if den else 1.0
    best = None
    for s0 in (0.2, 0.3, 0.45, 0.6, 0.8, 1.1):
        for c in (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.1):
            for q in (0.25, 0.35, 0.45, 0.55, 0.7):
                e = sum((max(0.3, s0 + c * greps[b] ** q) - sig[b]) ** 2 for b in range(nb))
                if best is None or e < best[0]:
                    best = (e, s0, c, q)
    _, s0, c, q = best
    wfit = {}
    for r in range(1, 11):
        bestw = None
        for dr in (0.85, 0.9, 0.94, 0.97, 1.0):
            for tau in (3, 5, 8, 12, 18, 26, 38, 55, 80, 120, 180):
                for p in (0.55, 0.7, 0.85, 1.0):
                    e = 0.0
                    for b in range(nb):
                        row = cells.get(r, {}).get(b)
                        if not row:
                            continue
                        tot = sum(row)
                        if tot < 2:
                            continue
                        e += ((dr * math.exp(-((greps[b] / tau) ** p))) - row[r] / tot) ** 2 * (tot / 96.0)
                    if bestw is None or e < bestw[0]:
                        bestw = (e, dr, tau, p)
        wfit[r] = bestw[1:]
    R = [rankmap(x) for x in reads]
    n_mean = sum(len(r) for r in R) / len(R)
    f_inf = min(0.10, 10.0 / max(n_mean, 1.0))
    surv = defaultdict(lambda: [[0, 0] for _ in range(nb)])
    for i, j, g in pairs(reads):
        b = bin_of(g)
        for u, rk in R[i].items():
            bb = band_of(rk)
            surv[bb][b][1] += 1
            if u in R[j] and R[j][u] <= 10:
                surv[bb][b][0] += 1
    mfit = {}
    for b in range(1, NB):
        bestm = None
        if b <= 10:
            grid = [(tau, p, 'd') for tau in (0.3, 0.6, 1, 2, 3.5, 6, 10, 16, 26, 40, 65, 100, 160, 250, 400)
                    for p in (0.3, 0.45, 0.6, 0.8, 1.0, 1.2)]
        else:
            grid = [(tau, p, 'r') for tau in (3, 8, 15, 25, 40, 65, 100, 150, 250)
                    for p in (0.5, 0.7, 0.9, 1.1, 1.3)]
        for tau, p, kind in grid:
            e = 0.0
            for k in range(nb):
                if greps[k] > 80:
                    continue
                hit, tot = surv[b][k]
                if tot < 2:
                    continue
                if kind == 'd':
                    pred = f_inf + (1 - f_inf) * math.exp(-((greps[k] / tau) ** p))
                else:
                    pred = f_inf * (1 - math.exp(-((greps[k] / tau) ** p)))
                e += (pred - hit / tot) ** 2 * (tot / 96.0)
            if bestm is None or e < bestm[0]:
                bestm = (e, tau, p, kind)
        mfit[b] = bestm[1:]
    # the model's clock is the hour; the curves are fitted on whatever spacing the history has
    k = 1.0
    return {'m': mfit, 'w': wfit, 'sig': (s0, c, q), 'k': k, 'beta': 1.0, 'a': WIDTH, 'f': f_inf}


def sigma_of(par, h):
    s0, c, q = par['sig']
    he = h / par['k']
    return max(0.3, (s0 + c * he ** q) * par['a'])


def m_of(par, b, h):
    tau, p, kind = par['m'][b]
    f = par['f']
    tau = min(tau, TAU_CAP) if kind == 'd' else tau
    he = h / par['k'] if b <= 10 else h
    if kind == 'd':
        return max(1e-9, min(1.0, f + (1 - f) * math.exp(-((he / tau) ** p))))
    return max(1e-9, min(1.0, f * (1 - math.exp(-((he / tau) ** p)))))


def w_of(par, r, h):
    dr, tau, p = par['w'][r]
    he = h / par['k']
    w = dr * math.exp(-((he / tau) ** p))
    return 1.0 - (1.0 - min(1.0, w)) ** par['beta']


def candidates(reads):
    last = rankmap(reads[-1])
    cands = dict(last)
    for row in reads[-5:-1]:
        for u, rk in rankmap(row).items():
            if u not in cands:
                cands[u] = min(97, rk + 25)
    return cands


def hour_marginals(reads, par, window_hours):
    cands = candidates(reads)
    out = {}
    for u, r in cands.items():
        b = band_of(r)
        q = [0.0] * PLACES
        for h in range(1, window_hours + 1):
            m = m_of(par, b, h)
            K = gauss_kernel(r, sigma_of(par, h))
            if r <= 10:
                w = w_of(par, r, h)
                for k in range(PLACES):
                    q[k] += m * (w * (1.0 if k == r - 1 else 0.0) + (1 - w) * K[k])
            else:
                for k in range(PLACES):
                    q[k] += m * K[k]
        out[u] = [x / window_hours for x in q]
    return out, len(cands)


def emit(q_hour, n_chart, resid=RESID, max_rows=MAX_ROWS):
    """Name each place's best games, keeping the rest of the place's mass for the games not named."""
    res = {}
    tot = {u: sum(q) for u, q in q_hour.items()}
    ids = sorted(q_hour, key=lambda u: (-tot[u], u))
    for k in range(PLACES):
        z = sum(q_hour[u][k] for u in ids) or 1.0
        best = None
        for keep in range(3, min(len(ids), max_rows) + 1):
            s = sum(q_hour[u][k] / z for u in ids[:keep])
            unnamed = max(1, n_chart - keep)
            u_mass = max(0.0, 1 - s * (1 - resid)) / unnamed
            gain = sum((q_hour[u][k] / z) * math.log(max(q_hour[u][k] / z, FLOOR)) for u in ids[:keep]) \
                + (1 - s) * math.log(max(u_mass, FLOOR))
            if best is None or gain > best[0]:
                best = (gain, keep, s)
        keep, s = best[1], best[2]
        scale = (1 - resid) / s if s > (1 - resid) else 1.0
        res[k + 1] = [(u, max(FLOOR, q_hour[u][k] / z * scale)) for u in ids[:keep]]
    return res


def top10_of(q_hour, rows):
    """Best guess at the top 10: the highest-probability game per place, kept distinct."""
    used = set()
    out = []
    for k in range(1, PLACES + 1):
        ranked = sorted(rows[k], key=lambda x: (-x[1], x[0]))
        pick = None
        for u, _ in ranked:
            if u not in used:
                pick = u
                break
        if pick is None:
            for u, _ in sorted(q_hour.items(), key=lambda x: -sum(x[1])):
                if u not in used:
                    pick = u
                    break
        used.add(pick)
        out.append(pick)
    return out


def main():
    task = json.load(open(os.path.join(INPUT_DIR, 'task.json')))
    places = int(task.get('places', PLACES))
    window = task.get('window') or {}
    window_hours = int(window.get('slots', 72))
    out_name = task.get('output', 'forecast.json')
    lines = [l for l in open(os.path.join(INPUT_DIR, 'history.ndjson'), encoding='utf-8').read().split('\n') if l.strip()]
    reads = [r for r in (json.loads(l) for l in lines) if r.get('ok') and r.get('up_and_coming')]
    if not reads:
        reads = [json.loads(l) for l in lines if l.strip()][-1:]
    par = fit(reads)
    q_hour, n_chart = hour_marginals(reads, par, window_hours)
    rows = emit(q_hour, n_chart)
    probabilities = []
    for k in range(1, min(places, PLACES) + 1):
        for u, p in rows[k]:
            probabilities.append({'universe_id': u, 'place': k, 'p': round(p, 7)})
    top = top10_of(q_hour, rows)[:places]
    with open(os.path.join(OUTPUT_DIR, out_name), 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'probabilities': probabilities, 'top10': top}, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
