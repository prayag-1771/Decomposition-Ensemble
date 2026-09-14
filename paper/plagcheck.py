"""Word n-gram overlap between the paper text and every cited title and
abstract in refs_raw.json. Reports the share of the paper's n-grams that also
occur in a source, and lists the shared 7-grams so each can be judged.

    python plagcheck.py body_draft.tex
"""
import json
import re
import sys

TEX = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
BS = chr(92)

tex = open(TEX, encoding='utf-8').read()
# the reference list quotes titles verbatim by design; check the prose only
tex = tex.split(BS + 'begin{thebibliography}')[0]
tex = re.sub(r'%.*', ' ', tex)
tex = re.sub(re.escape(BS) + r'(cite|label|ref|eqref)\{[^}]*\}', ' ', tex)
tex = re.sub(r'\$[^$]*\$', ' ', tex)
tex = re.sub(re.escape(BS) + r'[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', tex)


def words(s):
    return re.findall(r'[a-z]+', s.lower())


body = words(tex)
R = json.load(open('refs_raw.json', encoding='utf-8'))
src = {d: words((r.get('title') or '') + ' . ' + (r.get('abstract') or '')) for d, r in R.items()}

for n in (6, 7, 8, 10):
    grams = set(tuple(body[i:i + n]) for i in range(len(body) - n + 1))
    hits = {}
    for d, w in src.items():
        common = grams & set(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
        if common:
            hits[d] = sorted(' '.join(g) for g in common)
    shared = set(g for v in hits.values() for g in v)
    print('%2d-grams: %d of %d shared with a cited title or abstract (%.2f%%)'
          % (n, len(shared), len(grams), 100.0 * len(shared) / max(len(grams), 1)))
    if n == 7:
        for d, v in hits.items():
            print('     %s: %s' % (d, v))
