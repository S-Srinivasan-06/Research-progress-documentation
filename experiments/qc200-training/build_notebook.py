"""Build a self-contained code notebook without EEG, identities or credentials."""
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).parent

def main():
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        paths=list(ROOT.glob("*.py"))+list((ROOT/"repo").rglob("*.py"))
        paths+=[ROOT/"repo/data/electrode_graph/adj_mx_3d.pkl"]
        paths+=list(ROOT.glob("*.md"))+[ROOT/"config.B1.example.json"]
        for path in sorted(paths): bundle.write(path,path.relative_to(ROOT).as_posix())
    encoded=base64.b64encode(archive.getvalue()).decode("ascii")
    cells=[]
    def markdown(text): cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
    def code(text): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(True)})
    markdown("# Controlled 200 Hz DGDCN training\n\nThis notebook contains code only. It adapts the released model to TUSZ v2.0.6, retains official partitions, and completes training with dev results. Existing eval results were inspected previously. No new improvement is established.\n\nUse your existing runtime. Verify cache provenance and runtime paths before launch. B1 trains with unweighted BCE; W1 is a separate controlled loss-weight experiment.")
    code("from pathlib import Path\nfrom google.colab import drive\nimport os, sys, subprocess\nif not os.path.ismount('/content/drive'):\n    drive.mount('/content/drive')\nassert os.path.ismount('/content/drive'), 'Drive must be mounted for checkpoint persistence'\nsubprocess.run([sys.executable,'-m','pip','install','h5py','scipy','scikit-learn'],check=True)\nimport torch\nprint('CUDA available:',torch.cuda.is_available())\nif torch.cuda.is_available(): print(torch.cuda.get_device_name(0))\n")
    code("import base64, io, zipfile\nCODE=Path('/content/qc200-code')\nCODE.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode("+repr(encoded)+"))) as archive:\n    for entry in archive.infolist():\n        target=(CODE/entry.filename).resolve()\n        assert target.is_relative_to(CODE.resolve())\n    archive.extractall(CODE)\nprint('Active code restored')\n")
    code("# These paths and provenance are explicit inputs, not inferred runtime facts.\nimport json\nCONFIG=Path('/content/qc200-B1.json')\nconfig=json.loads((CODE/'config.B1.example.json').read_text())\n# Set preproc_dir to the existing extracted feature cache and marker_dir to the QC marker view.\n# Keep feature_contract_approved false until source/cache agreement and provenance are reviewed.\nprint(json.dumps(config,indent=2))\nCONFIG.write_text(json.dumps(config,indent=2))\n")
    code("# Run only after editing CONFIG with verified inputs. This audits every supplied feature.\nsubprocess.run([sys.executable,str(CODE/'preprocess_qc200.py'),'--config',str(CONFIG)],check=True)\n")
    code("# Training refuses incomplete validation or unapproved provenance.\n# Epoch and periodic checkpoints persist directly to the configured mounted Drive run folder.\nsubprocess.run([sys.executable,str(CODE/'train_qc200.py'),'--config',str(CONFIG)],check=True)\n")
    markdown("After B1 completes, compare its saved configuration and normalization before preparing W1. Select variants and thresholds using dev only. Final evaluation requires a separate frozen selection record. Event metrics require a validated continuous-interval scoring protocol; clip recall is not event sensitivity.")
    for index,cell in enumerate(cells): cell["id"]=f"cell-{index:04d}"
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":cells}
    (ROOT/'QC200_Training.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print('Wrote code-only QC200_Training.ipynb')

if __name__=='__main__': main()
