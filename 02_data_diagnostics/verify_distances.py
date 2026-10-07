"""Cross-check Matrix_Jarak_Jawa_Timur_simetris.xlsx (compiled by the author with an LLM search)
against OpenStreetMap road routing (OSRM), between the administrative seats of the 38 regencies
and cities.  Read-only; writes distance_check.csv and distance_check.json next to this script.

Seats: a regency's capital town, a city's own name.  Geocoding with Nominatim (1 request/s),
routing with the public OSRM table service (driving distance in km).
"""
import json, os, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {'User-Agent': 'cmin-distance-check/1.0'}
D = pd.read_excel(os.path.join(HERE, 'Matrix_Jarak_Jawa_Timur_simetris.xlsx'))
labels = list(D.iloc[:, 0])
M = D.iloc[:, 1:].to_numpy(float)

SEAT = {'Pacitan': 'Pacitan', 'Ponorogo': 'Ponorogo', 'Trenggalek': 'Trenggalek', 'Tulungagung': 'Tulungagung',
        'Blitar Kab.': 'Kanigoro, Blitar', 'Kediri Kab.': 'Ngasem, Kediri', 'Malang Kab.': 'Kepanjen',
        'Lumajang': 'Lumajang', 'Jember': 'Jember', 'Banyuwangi': 'Banyuwangi', 'Bondowoso': 'Bondowoso',
        'Situbondo': 'Situbondo', 'Probolinggo Kab.': 'Kraksaan', 'Pasuruan Kab.': 'Bangil',
        'Sidoarjo': 'Sidoarjo', 'Mojokerto Kab.': 'Mojosari', 'Jombang': 'Jombang', 'Nganjuk': 'Nganjuk',
        'Madiun Kab.': 'Mejayan, Madiun', 'Magetan': 'Magetan', 'Ngawi': 'Ngawi', 'Bojonegoro': 'Bojonegoro',
        'Tuban': 'Tuban', 'Lamongan': 'Lamongan', 'Gresik': 'Gresik', 'Bangkalan': 'Bangkalan',
        'Sampang': 'Sampang', 'Pamekasan': 'Pamekasan', 'Sumenep': 'Sumenep', 'Kota Kediri': 'Kota Kediri',
        'Kota Blitar': 'Kota Blitar', 'Kota Malang': 'Kota Malang', 'Kota Probolinggo': 'Kota Probolinggo',
        'Kota Pasuruan': 'Kota Pasuruan', 'Kota Mojokerto': 'Kota Mojokerto', 'Kota Madiun': 'Kota Madiun',
        'Kota Surabaya': 'Surabaya', 'Kota Batu': 'Kota Batu'}

cache_f = os.path.join(HERE, 'distance_geocode_cache.json')
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
coords = []
for lab in labels:
    q = SEAT[lab] + ', Jawa Timur, Indonesia'
    if q not in cache:
        url = 'https://nominatim.openstreetmap.org/search?' + urllib.parse.urlencode({'q': q, 'format': 'json', 'limit': 1})
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        cache[q] = [float(r[0]['lat']), float(r[0]['lon']), r[0]['display_name']] if r else None
        json.dump(cache, open(cache_f, 'w'), indent=1)
        time.sleep(1.1)
    if cache[q] is None:
        raise SystemExit('not geocoded: ' + q)
    coords.append(cache[q])
    print('%-18s %9.4f %9.4f  %s' % (lab, cache[q][0], cache[q][1], cache[q][2][:60]))

pts = ';'.join('%.5f,%.5f' % (c[1], c[0]) for c in coords)
url = 'https://router.project-osrm.org/table/v1/driving/' + pts + '?annotations=distance'
T = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120))
O = np.array(T['distances'], float) / 1000.0
O = (O + O.T) / 2

off = ~np.eye(38, dtype=bool)
a, b = M[off], O[off]
from scipy import stats
dev = np.abs(a - b) / b
rows = []
for i in range(38):
    for j in range(i + 1, 38):
        rows.append(dict(a=labels[i], b=labels[j], matrix_km=M[i, j], osrm_km=round(O[i, j], 1),
                         diff_km=round(M[i, j] - O[i, j], 1), rel_dev=round(abs(M[i, j] - O[i, j]) / O[i, j], 3)))
R = pd.DataFrame(rows).sort_values('rel_dev', ascending=False)
R.to_csv(os.path.join(HERE, 'distance_check.csv'), index=False)


def knn_sets(X, k=8):
    return [set(np.argsort(np.where(np.eye(38, dtype=bool)[i], np.inf, X[i]))[:k]) for i in range(38)]


kn_m, kn_o = knn_sets(M), knn_sets(O)
overlap = np.mean([len(kn_m[i] & kn_o[i]) / 8 for i in range(38)])
summary = dict(pearson_r=float(stats.pearsonr(a, b)[0]), spearman_rho=float(stats.spearmanr(a, b)[0]),
               median_rel_dev=float(np.median(dev)), p90_rel_dev=float(np.quantile(dev, 0.9)),
               median_abs_km=float(np.median(np.abs(a - b))), mean_ratio_matrix_over_osrm=float(np.mean(a / b)),
               knn8_overlap=float(overlap), pairs=int(len(a) / 2))
json.dump(summary, open(os.path.join(HERE, 'distance_check.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
print(R.head(12).to_string(index=False))
