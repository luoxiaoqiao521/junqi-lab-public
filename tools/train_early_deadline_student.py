"""Train and run a student that chooses directly without a teacher rollout."""
import json
import os
import subprocess
import sys
import time

from compare_book_learning import OUTPUTS, fit, label_vector, np
from evidence_model import sha
from export_deadline_learning_pairs import identities
from measure_deadline_opening import ROOT, RULES, build, measure, model_roster, paired, require


SOURCE = (ROOT / "build/deadline-opening/export-early-deadline-teacher-pairs-3838208c4a548ef5b52d"
          / "early-public.jsonl")
ENCODING = "junqi-public-early-action-v1"
EXCHANGE = "junqi-public-early-action-collision-v1"
WIDTH, FEATURES, SEED, COUNT = 24, 41, 1093400, 24
COLUMNS = ("seed", "seat", "partner", "table", "plies", "outcome", "changed", "opportunities",
           "trace", "first", "friendly_retired", "enemy_retired")


def native(directory, mode, text):
    t0 = time.perf_counter()
    process = subprocess.run([directory/"deadline_opening.exe", RULES, mode], input=text, capture_output=True,
                             encoding="utf8", errors="replace", timeout=170)
    require(process.returncode==0, f"{mode} failed: {process.stderr.strip()}")
    print(f"{mode} seconds={time.perf_counter()-t0:.3f}")
    return process.stdout


def training(directory):
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf8").splitlines()]
    groups = {}
    training = []
    for row in rows:
        require(row["version"]=="junqi-early-deadline-outcomes-1", "incompatible source")
        group = row["partition"]
        require(groups.setdefault(row["family"], group)==group, "layout split crossed")
        if group=="development_train":
            training.append(row)
    return fit_student(directory, training, (("linear",0),("small_nn",WIDTH)), sha(SOURCE))


def mean_label(branch):
    if "reply_outcomes" not in branch:
        return label_vector(branch["outcome"])
    samples=np.asarray([label_vector(label) for label in branch["reply_outcomes"]])
    counts=np.isfinite(samples).sum(axis=0)
    return np.divide(np.nansum(samples,axis=0),counts,out=np.full(samples.shape[1],np.nan),where=counts>0)


def fit_student(directory, training, models, source, parent=None, exchange=False):
    payload = [str(len(training))]
    labels = []
    for row in training:
        state = row["observation"]
        fields = [state["actor"],state["ply"],state["static_plies"],len(state["legal"])]
        identities(state["board"],state["actor"])
        require(len(state["board"])==129, "invalid board width")
        fields.extend(v for node in state["board"] for v in node)
        fields.extend(state["legal"])
        fields.append(len(row["branches"]))
        for move,branch in row["branches"].items():
            fields.append(int(move))
            labels.append(mean_label(branch))
        payload.append(" ".join(map(str,fields)))
    mode = "student-features-exchange" if exchange else "student-features"
    dims = FEATURES+3 if exchange else FEATURES
    wire = native(directory,mode,"\n".join(payload))
    x = np.asarray([list(map(float,line.split())) for line in wire.splitlines()])
    y = np.asarray(labels)
    require(x.shape==(len(y),dims) and np.isfinite(x).all(), "unaligned native encoding")
    if parent is not None:
        parent_artifact,parent_name=parent
        model_arguments(parent_artifact,parent_name)
        mean,scale=np.asarray(parent_artifact["mean"]),np.asarray(parent_artifact["scale"])
        initial=parent_artifact["models"][parent_name]
        require(mean.shape==scale.shape==(FEATURES,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale>0).all(),"incompatible parent normalization")
    else:
        mean = x.mean(axis=0)
        scale = np.where(x.std(axis=0)>.025,x.std(axis=0),1)
        initial=None
    if exchange and parent:
        require(len(models)==1 and models[0][1]>0,"exchange warm start needs one hidden layer")
        mean = np.append(mean,np.zeros(3))
        scale = np.append(scale,np.ones(3))
        initial = dict(initial,weights=[initial["weights"][0]+[[0.]*models[0][1] for _ in range(3)],
                                       *initial["weights"][1:]])
    normal = np.clip((x-mean)/scale,-6,6)
    return {"encoding":EXCHANGE if exchange else ENCODING, "rules":sha(RULES), "source":source, "outputs":list(OUTPUTS),
            "mean":mean.tolist(), "scale":scale.tolist(), "roots":len(training), "labels":len(labels),
            "models":{name:fit(normal,y,17,width,initial) for name,width in models}}


def model_arguments(artifact, name, encoding=ENCODING):
    require(encoding in (ENCODING,EXCHANGE) and artifact["encoding"]==encoding and artifact["rules"]==sha(RULES)
            and artifact["outputs"]==list(OUTPUTS),"incompatible student")
    model = artifact["models"][name]
    weights,bias = model["weights"],model["bias"]
    width = len(bias[0]) if len(weights)==2 else 0
    hidden = [x for row in weights[0] for x in row]+bias[0] if width else []
    dims = FEATURES+3 if encoding==EXCHANGE else FEATURES
    values = [encoding,dims,len(OUTPUTS),width,*artifact["mean"],*artifact["scale"],
              *hidden,*(x for row in weights[-1] for x in row),*bias[-1]]
    return " ".join(map(str,values))


def evaluation(directory, artifact, seed=SEED, count=COUNT, contact_detail=False):
    roster = model_roster()
    values = [seed,count,*(w for n in ("candidate","historical") for w in roster[n]["weights"])]
    prefix = " ".join(map(str,values))
    result = {}
    grid = [(s,j,p,t) for s in range(seed,seed+count) for j in range(4) for p in (1,2) for t in (0,1)]
    artifacts = artifact if isinstance(artifact, tuple) else (artifact,)
    contestants = {name:fit for fit in artifacts for name in fit["models"]}
    require("pressure" not in contestants, "reused pressure arm")
    for arm in ("pressure",*contestants):
        encoding=ENCODING if arm=="pressure" else contestants[arm]["encoding"]
        source = prefix if arm=="pressure" else prefix+" "+model_arguments(contestants[arm],arm,encoding)
        mode="student-reference" if arm=="pressure" else "student-apply"
        if encoding==EXCHANGE:
            require(contact_detail,"exchange audit requires contact telemetry")
            mode+="-exchange"
        elif contact_detail:
            mode+="-contact"
        output = native(directory,mode,source)
        columns=COLUMNS+("legal_contact","shortlisted_contact","selected_contact","attacker_wins") if contact_detail else COLUMNS
        rows = [dict(zip(columns,map(int,line.split()),strict=True)) for line in output.splitlines()]
        require([(r["seed"],r["seat"],r["partner"],r["table"]) for r in rows]==grid
                and all(0<=r["changed"]<=r["opportunities"]<=2 for r in rows),"broken student replay grid")
        for row in rows:
            if contact_detail:
                require(0<=row["attacker_wins"]<=row["selected_contact"]<=row["shortlisted_contact"]
                        <=row["legal_contact"]<=2,"invalid contact ledger")
            row["changed100"]=row["changed140"]=row["changed"]
        result[arm] = rows
    verify_codecs(directory,contestants,prefix)
    return result


def verify_codecs(directory,contestants,prefix):
    test_arm = next(arm for arm in contestants if contestants[arm]["encoding"]==ENCODING)
    invalid = prefix+" "+model_arguments(contestants[test_arm],test_arm).replace(ENCODING,"junqi-wrong-encoding",1)
    reject = subprocess.run([directory/"deadline_opening.exe",RULES,"student-apply"],input=invalid,
                            capture_output=True,text=True,timeout=10)
    require(reject.returncode==2 and not reject.stdout,"incompatible encoding selected an action")
    for other,model in contestants.items():
        if model["encoding"]!=EXCHANGE:
            continue
        for path,wire in (("student-apply",model_arguments(model,other,EXCHANGE)),
                          ("student-apply-exchange",model_arguments(contestants[test_arm],test_arm))):
            invalid=subprocess.run([directory/"deadline_opening.exe",RULES,path],input=prefix+" "+wire,
                                   capture_output=True,text=True,timeout=10)
            require(invalid.returncode==2 and not invalid.stdout,"cross-encoding weight load accepted")


def panels(arms):
    report = {}
    for partner in (1,2):
        for table in (0,1):
            groups = {name:[r for r in rows if r["partner"]==partner and r["table"]==table]
                      for name,rows in arms.items()}
            before = measure(groups["pressure"])
            cell = {"pressure":before}
            for name in (name for name in groups if name!="pressure"):
                after = measure(groups[name])
                delta = paired(groups["pressure"],groups[name],before,after)
                actual = list(zip(groups["pressure"],groups[name],strict=True))
                require(all(b["changed"] or a["trace"]==b["trace"] for a,b in actual),"unchanged choices diverged")
                cell[name] = after | {"delta":delta,"actual_changes":sum(b["changed"] for _,b in actual),
                                     "different_games":sum(a["trace"]!=b["trace"] for a,b in actual)}
            report[f"{partner}/{table}"] = cell
    return report


def main(audit):
    directory = ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    artifact = training(directory)
    model_wire = json.dumps(artifact,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n"
    model_path = directory/"student.json"
    if audit:
        require(model_path.read_bytes()==model_wire.encode(),"student weights did not reproduce")
    else:
        pending = model_path.with_suffix(".tmp")
        pending.write_bytes(model_wire.encode())
        pending.replace(model_path)
    arms = evaluation(directory,json.loads(model_path.read_bytes()))
    report = panels(arms)
    data = {"training_roots":artifact["roots"],"training_labels":artifact["labels"],
            "rules":artifact["rules"],"result":report,"games":arms,"model_bytes":len(model_wire.encode())}
    wire = json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n"
    receipt = directory/"summary.json"
    if audit:
        require(receipt.read_bytes()==wire.encode(),"student outcomes did not reproduce")
    else:
        pending = receipt.with_suffix(".tmp")
        pending.write_bytes(wire.encode())
        pending.replace(receipt)
    print(json.dumps(report,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
