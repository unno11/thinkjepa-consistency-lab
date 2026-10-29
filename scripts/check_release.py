"""Check public package paths, frozen manifests and archived result identity."""
from pathlib import Path
from urllib.parse import unquote
import ast,json,re,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def check():
    errors=[];python_files=0
    for p in ROOT.rglob('*.py'):
        if any(x in p.parts for x in ['.git','__pycache__','.testtmp','.venv']):continue
        python_files+=1
        try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.relative_to(ROOT).as_posix())
        except (SyntaxError,UnicodeError) as exc:errors.append(str(exc))
    for p in ROOT.rglob('*.json'):
        if any(x in p.parts for x in ['.git','.testtmp','.venv']):continue
        try:json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:errors.append(str(exc))
    for p in [ROOT/'README.md',*(ROOT/'docs').rglob('*.md'),*(ROOT/'results').glob('*.md')]:
        text=p.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if re.match(r'https?://|mailto:|#',target):continue
            if not (p.parent/unquote(target.split('#')[0])).exists():errors.append(f'{p.name}: missing {target}')
    for name,expected in [('train','ceec4fad0984e31c0eb241b4b9dfdcb00e78a97a3af5f258c6605cf94fedec8e'),('val','82f237a5acbd99f078dd1dd8675bf66b041d0ce56c885a3141817d8ece401b15')]:
        p=ROOT/'protocol/splits'/f'{name}.txt'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append(f'Changed manifest: {name}')
    if (ROOT/'protocol/splits/test.txt').exists():errors.append('Final-test membership must not be shipped')
    required=['LICENSE','NOTICE','egodex/LICENSE.txt','egodex/ACKNOWLEDGEMENTS.txt','egodex/utils/LICENSE.txt','egodex/utils/ACKNOWLEDGEMENTS.txt','vjepa2/LICENSE','vjepa2/APACHE-LICENSE']
    for name in required:
        if not (ROOT/name).is_file():errors.append('Missing attribution: '+name)
    from scripts.summarize_results import summarize
    report,_=summarize()
    if errors:raise ValueError('\n'.join(errors))
    print(json.dumps({'python_files_parsed':python_files,'local_document_links':'passed','licenses':'present','archived_results':'verified','validation_samples_per_seed':report['validation_samples_per_seed'],'final_test_shipped':False},indent=2))
if __name__=='__main__':check()
