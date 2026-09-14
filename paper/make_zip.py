"""Build the Overleaf upload set: the overleaf/ folder and the same files zipped.

    python make_zip.py

overleaf/ holds exactly what a project needs and nothing else:
    main.tex               the manuscript (cites refs.bib through the class's style)
    refs.bib               generated from publisher records by make_refs.py
    sn-jnl.cls             official Springer Nature class, December 2024 package
    sn-mathphys-num.bst    matching bibliography style from the same package
    figs/*.png             the three figures
Overleaf: New Project -> Upload Project (the zip), or a blank project with
these files uploaded and figs/ created as a folder. Compile with pdfLaTeX.
main_selfcontained.tex in this folder is the same paper with the reference
list written out, for a submission system that wants one .tex file.
"""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'overleaf')
OUT_ZIP = os.path.join(HERE, 'Decomposition_Forecasting_Overleaf.zip')
files = {'main.tex': 'main.tex', 'refs.bib': 'refs.bib',
         'sn-jnl.cls': 'template/sn-jnl.cls',
         'sn-mathphys-num.bst': 'template/sn-mathphys-num.bst'}
for f in sorted(os.listdir(os.path.join(HERE, 'figs'))):
    if f.endswith('.png'):
        files['figs/' + f] = 'figs/' + f

shutil.rmtree(OUT_DIR, ignore_errors=True)
os.makedirs(os.path.join(OUT_DIR, 'figs'))
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for arc, src in files.items():
        shutil.copyfile(os.path.join(HERE, src), os.path.join(OUT_DIR, arc))
        z.write(os.path.join(HERE, src), arc)      # flat layout, forward slashes
print('overleaf/ and %s: %d files' % (os.path.basename(OUT_ZIP), len(files)))
for arc in files:
    print('  ' + arc)
