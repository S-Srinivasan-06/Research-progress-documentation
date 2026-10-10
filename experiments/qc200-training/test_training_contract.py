"""Small integration checks; synthetic data never represent seizure performance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import h5py
from run_contract import SOURCE_VERSION, sampler_statistics
from preprocess_qc200 import log_fft
from scipy.fftpack import fft

HERE=Path(__file__).parent

class TrainingContract(unittest.TestCase):
    def test_fft_equivalence(self):
        rng=np.random.RandomState(123)
        signal=rng.normal(size=(19,2400))
        expected=[]
        for step in range(12):
            magnitude=np.abs(fft(signal[:,step*200:(step+1)*200],n=200,axis=-1)[:,:100])
            magnitude[magnitude==0]=1e-8
            expected.append(np.log(magnitude))
        np.testing.assert_array_equal(log_fft(signal),np.asarray(expected,dtype=np.float32))
        self.assertTrue(np.isfinite(log_fft(np.zeros((19,2400)))).all())

    def test_sampler_prior(self):
        rows=[("a",1,"A"),("b",0,"A"),("c",0,"A"),("d",0,"B")]
        stats=sampler_statistics(rows)
        self.assertAlmostEqual(stats["positive_probability"],1/6)
        self.assertAlmostEqual(stats["candidate_pos_weight"],5)

    def test_train_dev_isolation_and_resume_guard(self):
        with tempfile.TemporaryDirectory(prefix="qc200-test-") as folder:
            root=Path(folder); markers=root/"markers"; markers.mkdir()
            features=root/"features"
            rng=np.random.RandomState(10)
            for split,patient in (("train","trainperson"),("dev","devperson"),("test","evalperson")):
                disk="eval" if split=="test" else split
                for kind,label in (("sz",1),("nosz",0)):
                    lines=[]
                    for index in range(2):
                        row_patient="trainother" if split=="train" and kind=="nosz" and index==1 else patient
                        rel=f"{disk}/{row_patient}/{kind}.edf_{index}.h5"
                        target=features/rel; target.parent.mkdir(parents=True,exist_ok=True)
                        with h5py.File(target,"w") as f:
                            f.create_dataset("clip",data=rng.normal(size=(12,19,100)).astype(np.float32))
                            f.attrs["feature_version"]="synthetic-v1"
                            f.attrs["producer_version"]=SOURCE_VERSION
                        lines.append(f"{rel}, {label}\n")
                    (markers/f"{split}Set_seq2seq_12s_{kind}.txt").write_text("".join(lines))
            env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
            def call(script,config,expect=0):
                path=root/"config.json"; path.write_text(json.dumps(config))
                result=subprocess.run([sys.executable,str(HERE/script),"--config",str(path)],
                                      capture_output=True,text=True,env=env,timeout=180)
                if expect==0 and result.returncode:
                    self.fail(result.stdout+result.stderr)
                if expect!=0: self.assertNotEqual(result.returncode,0)
                return result
            config={"run_id":"synthetic-B1","variant":"B1","feature_version":"synthetic-v1",
                    "feature_contract_approved":True,"preproc_dir":str(features),
                    "marker_dir":str(markers),"output_dir":str(root/"B1"),
                    "epochs":1,"batch_size":2,"eval_batch_size":2,"workers":0,"device":"cpu"}
            config["expected_partition_rows"]={"train":4,"dev":4,"test":4}
            config["cohort_kind"]="synthetic-test"
            call("preprocess_qc200.py",config)
            # After the full input audit, remove eval feature files. Training must never open them.
            # Call the audited low-level trainer after making eval H5 paths unavailable.
            # Public config wrapper intentionally rechecks the full inventory before launch.
            moved=features/"eval-disabled"
            (features/"eval").rename(moved)
            argv=[sys.executable,str(HERE/"trainer.py"),"--repo",str(HERE/"repo"),
                  "--preproc-dir",str(features),"--marker-dir",str(markers),"--output-dir",config["output_dir"],
                  "--run-id",config["run_id"],"--feature-version",config["feature_version"],
                  "--patient-balanced-sampler","--epochs","1","--batch-size","2",
                  "--eval-batch-size","2","--workers","0","--device","cpu",
                  "--threshold-objective","sensitivity"]
            result=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            moved.rename(features/"eval")
            output=root/"B1"
            self.assertEqual(json.loads((output/"run-status.json").read_text())["state"],"complete_train_dev")
            self.assertFalse((output/"final_eval_metrics.json").exists())
            self.assertFalse((output/"eval_access_log.jsonl").exists())
            self.assertTrue((output/"dev_predictions.csv.gz").exists())
            call("finalize_qc200.py",config,expect=1)
            config["resume"]=str(output/"last.pt")
            call("train_qc200.py",config)
            config["variant"]="W1"
            failed=call("train_qc200.py",config,expect=1)
            self.assertIn("differs",failed.stderr)
            config.pop("resume")
            config["run_id"]="synthetic-W1"; config["output_dir"]=str(root/"W1")
            call("preprocess_qc200.py",config)
            call("train_qc200.py",config)
            saved=json.loads((root/"W1/config.json").read_text())
            self.assertAlmostEqual(saved["training_pos_weight"],2)
            self.assertEqual(saved["dev_loss"],"unweighted BCE")
            config.update(run_id="synthetic-B1",variant="B1",output_dir=str(output))
            config["checkpoint_destination"]=str(root/"persistent")
            (root/"persistent").mkdir()
            call("checkpoint_sync.py",config)
            self.assertEqual((root/"persistent/synthetic-B1/last.pt").read_bytes(),(output/"last.pt").read_bytes())
            selection=root/"selection.json"
            dev=json.loads((output/"dev_metrics.json").read_text())
            selection.write_text(json.dumps({"run_id":"synthetic-B1","selected_on":"dev",
                "checkpoint_epoch":1,"threshold":dev["threshold"],"frozen_at_utc":"synthetic-test",
                "previous_eval_access_disclosed":True}))
            config["selection_record"]=str(selection)
            call("finalize_qc200.py",config)
            self.assertEqual(json.loads((output/"final_eval_metrics.json").read_text())["count"],4)
            call("finalize_qc200.py",config,expect=1)

if __name__=="__main__": unittest.main()
