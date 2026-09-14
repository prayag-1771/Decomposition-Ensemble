"""Build refs.tex (thebibliography entries) and REFERENCES.md from the
publisher records in refs_raw.json (Crossref; DataCite for arXiv).

Nothing is typed by hand except fields a record lacks or states in a form
unsuitable for print; each such field is in OVERRIDES with its source.
Entries are ordered by first citation in the given .tex file, and the script
fails if a cited key has no record, a record is never cited, or any cited work
was published before MIN_YEAR.

    python make_refs.py body_draft.tex
"""
import html
import json
import re
import sys
import textwrap

TEX = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
MIN_YEAR = 2022
PRIMARY_SOURCES = {'colominas2014', 'dragomiretskiy2014'}   # exempt from MIN_YEAR
R = json.load(open('refs_raw.json', encoding='utf-8'))

KEYS = {
    # the model reproduced and its recursive head
    'gong2024': '10.1016/j.asoc.2024.112393',
    'jolicoeur2025': '10.48550/arxiv.2510.04871',
    # decomposition-ensemble forecasting
    'akimaemd2023': '10.3390/app13031429',
    'gruceemdan2023': '10.3390/app13127104',
    'ceemdanstock2025': '10.1155/jama/7706431',
    'wangwang2026': '10.1007/s10614-025-11148-z',
    'jiang2025': '10.3390/app152011169',
    'ning2025': '10.3390/pr13072046',
    'zhao2025': '10.3390/math13162622',
    'damasevicius2024': '10.7717/peerj-cs.1795',
    'psovmdtcn2023': '10.3390/en16124616',
    'kreuzer2025': '10.1007/s10618-025-01120-8',
    # decomposition methods (the two 2014 entries are the primary sources of
    # ICEEMDAN and VMD, the methods this study runs; the rest is 2022 or later)
    'colominas2014': '10.1016/j.bspc.2014.06.009',
    'dragomiretskiy2014': '10.1109/tsp.2013.2288675',
    'emdtutorial2023': '10.1109/access.2023.3307628',
    'ceemdanpaa2022': '10.3390/s22176599',
    'iceemdanbridge2022': '10.1002/stc.2966',
    'vmdmssr2022': '10.1088/1361-6501/ac8c63',
    'psosurvey2022': '10.1109/access.2022.3142859',
    # leakage and future information
    'chen2022': '10.1016/j.egyr.2022.07.005',
    'wang2024': '10.1016/j.energy.2024.133513',
    'yao2025': '10.3390/pr13030819',
    'pang2026': '10.1016/j.apm.2025.116376',
    'emdleak2024': '10.1038/s41598-024-80018-9',
    'kapoor2023': '10.1016/j.patter.2023.100804',
    'glasserman2024': '10.3905/jfds.2023.1.143',
    # evaluation, benchmarks and networks
    'kumbure2022': '10.1016/j.eswa.2022.116659',
    'olorunnimbe2022': '10.1007/s10462-022-10226-0',
    'makridakis2022': '10.1080/01605682.2022.2118629',
    'finrel2024': '10.1186/s40854-024-00644-0',
    'pernagallo2025': '10.1007/s11135-025-02052-7',
    'hewamalage2022': '10.1007/s10618-022-00894-5',
    'mdmindex2023': '10.1016/j.nexus.2023.100210',
    'zeng2023': '10.1609/aaai.v37i9.26317',
    'rnnreview2024': '10.3390/info15090517',
}

OVERRIDES = {}

ACCENTS = {'š': r'\v{s}', 'č': r'\v{c}', 'ć': r"\'{c}", 'é': r"\'{e}", 'á': r"\'{a}",
           'í': r"\'{\i}", 'ó': r"\'{o}", 'ú': r"\'{u}", 'ü': r'\"{u}', 'ö': r'\"{o}',
           'ñ': r'\~{n}', 'ş': r'\c{s}', 'ç': r'\c{c}', 'ã': r'\~{a}', 'è': r'\`{e}'}


def clean(s):
    s = html.unescape(re.sub(r'<[^>]+>', '', str(s)))
    return re.sub(r'\s+', ' ', s).strip()


def latex(s):
    s = s.replace('&', r'\&')
    for ch, rep in ACCENTS.items():
        s = s.replace(ch, rep)
    return s


def initials(given):
    out = []
    for part in given.replace('.', '. ').split():
        subs = [p for p in part.split('-') if p]
        out.append('-'.join(p[0] + '.' for p in subs))
    return ' '.join(out)


def name(raw, datacite):
    raw = clean(raw)
    if datacite and ', ' in raw:
        family, given = raw.split(', ', 1)
    elif ' ' in raw:
        toks = raw.split()
        fam = [toks.pop()]
        # keep lower-case particles with the family name: van Jaarsveldt
        while len(toks) > 1 and toks[-1] in ('van', 'von', 'de', 'der', 'den', 'da', 'di', 'du', 'le', 'la'):
            fam.insert(0, toks.pop())
        given, family = ' '.join(toks), ' '.join(fam)
    else:
        return raw
    return '%s %s' % (initials(given), family)


def record(key):
    r = dict(R[KEYS[key]])
    r.update(OVERRIDES.get(key, {}))
    return r


def entry(key):
    r = record(key)
    doi = KEYS[key]
    arxiv = doi.startswith('10.48550')
    names = [name(a, arxiv) for a in r['authors']]
    au = names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]
    title = clean(r['title'])
    if arxiv:
        tail = 'arXiv preprint arXiv:%s, %s.' % (doi.split('arxiv.')[1], r['year'])
    else:
        bits = [clean(r['venue'])]
        vol, iss = r.get('volume'), r.get('issue')
        if vol:
            bits.append('vol. %s' % (str(int(vol)) if str(vol).isdigit() else vol))
        if iss:
            bits.append('no. %s' % (str(int(iss)) if str(iss).isdigit() else iss))
        page, art = r.get('page'), r.get('article')
        if page and '-' in page:
            bits.append('pp. %s' % page)
        elif art or page:
            bits.append('Art. no. %s' % (art or page))
        bits.append(str(r['year']))
        tail = ', '.join(bits) + '.'
    comma = '' if title.endswith(('?', '!')) else ','
    text = latex('%s, ``%s%s\'\' %s' % (au, title, comma, tail))
    body = textwrap.fill(text, 86, break_on_hyphens=False, break_long_words=False)
    return '\\bibitem{%s} %s' % (key, body), au, title, r['year'], doi


tex = open(TEX, encoding='utf-8').read()
order = []
for group in re.findall(r'\\cite\{([^}]*)\}', tex):
    for k in group.split(','):
        k = k.strip()
        if k and k not in order:
            order.append(k)
missing = [k for k in order if k not in KEYS]
unused = [k for k in KEYS if k not in order]
if missing:
    sys.exit('cited but no record: %s' % missing)
if unused:
    sys.exit('record never cited: %s' % unused)
old = ['%s (%s)' % (k, record(k)['year']) for k in order
       if int(record(k)['year']) < MIN_YEAR and k not in PRIMARY_SOURCES]
if old:
    sys.exit('published before %d: %s' % (MIN_YEAR, old))

def bibsafe(s):
    """Wrap accent commands in braces so BibTeX keeps them whole."""
    return re.sub(r'(\\[a-zA-Z\'`"^~=]\{[^}]*\})', r'{\1}', s)


def bibentry(key):
    """One BibTeX record from the same publisher fields as the printed entry."""
    r = record(key)
    doi = KEYS[key]
    arxiv = doi.startswith('10.48550')
    names = []
    for a in r['authors']:
        a = clean(a)
        if arxiv and ', ' in a:
            fam, giv = a.split(', ', 1)
            a = giv + ' ' + fam
        names.append(bibsafe(latex(a)))
    fields = [('author', ' and '.join(names)), ('title', '{' + bibsafe(latex(clean(r['title']))) + '}')]
    if arxiv:
        kind = 'article'
        fields.append(('journal', 'arXiv preprint arXiv:' + doi.split('arxiv.')[1]))
    else:
        kind = 'inproceedings' if r.get('type') == 'proceedings-article' else 'article'
        fields.append(('booktitle' if kind == 'inproceedings' else 'journal', latex(clean(r['venue']))))
        if r.get('volume'):
            fields.append(('volume', str(r['volume'])))
        if r.get('issue'):
            fields.append(('number', str(r['issue'])))
        page, art = r.get('page'), r.get('article')
        if page and '-' in page:
            fields.append(('pages', page.replace('-', '--')))
        elif art or page:
            fields.append(('pages', str(art or page)))
    fields += [('year', str(r['year'])), ('doi', doi)]
    body = (',' + chr(10)).join('  %s = {%s}' % (f, v) for f, v in fields)
    return ('@%s{%s,' + chr(10) + '%s' + chr(10) + '}') % (kind, key, body)


tex_out, bib_out, md = [], [], ['| No. | Reference | Year | Link |', '|---:|---|---:|---|']
for i, k in enumerate(order, 1):
    item, au, title, year, doi = entry(k)
    tex_out.append(item)
    bib_out.append(bibentry(k))
    md.append('| %d | %s, "%s" | %s | https://doi.org/%s |' % (i, au, title, year, doi))
open('refs.bib', 'w', encoding='utf-8', newline=chr(10)).write(
    '% Generated by make_refs.py from publisher records (refs_raw.json); do not edit by hand.'
    + chr(10) * 2 + (chr(10) * 2).join(bib_out) + chr(10))
open('refs.tex', 'w', encoding='utf-8', newline='\n').write(
    '\\begin{thebibliography}{99}\n\n' + '\n\n'.join(tex_out) + '\n\n\\end{thebibliography}\n')
open('REFERENCES.md', 'w', encoding='utf-8', newline='\n').write(
    '# References\n\nEvery entry is built from its publisher record (Crossref, or DataCite '
    'for arXiv) by make_refs.py. All were published in %d or later, except the two\n'
    'primary sources of the decomposition methods used (2014).\n\n' % MIN_YEAR
    + '\n'.join(md) + '\n')
print('%d references, %d from %d or later plus %d primary method sources (refs.tex, refs.bib, REFERENCES.md)'
      % (len(order), sum(1 for k in order if int(record(k)['year']) >= MIN_YEAR), MIN_YEAR,
         sum(1 for k in order if k in PRIMARY_SOURCES)))
