"""Test one predeclared post-exchange restriction on fresh complete games."""
from collections import Counter
from itertools import product
import json
import os
import sys

from audit_exchange_first_reply import MAX_EXTRA_LOSS, MIN_GAIN, PARENT, panel, parse, verify_parent
from cross_partner_deadline_student import encode, persist
from measure_deadline_opening import ROOT, build, model_roster, require
from train_early_deadline_student import model_arguments, native

START, COUNT = 1096300, 48
CONTEXT = ("contact_ply","contact_outcome","next_ply","choices","base","prefix","replies",
           "survives","first_reply","ally_contact","enemy_contacts")


def check_gated(broad, gated):
    counts = Counter()
    for key, case in broad.items():
        probe = gated[key]
        parent = case["runs"]["0/0"]
        p,q = case.get("pivot"),probe.get("pivot")
        require(case["first"]==probe["first"] and parent==probe["runs"]["0/0"],
                "restriction changed the parent's trajectory")
        if p is None:
            require(q is None and case==probe,"restriction changed an untriggered game")
            counts["no_pivot"]+=1
            continue
        require(q and all(q[k]==p[k] for k in CONTEXT),"restriction changed the common observation")
        if p["survives"]:
            require(q["method"]==5 and q["samples"]==0 and q["alternate"]==p["base"]
                    and probe["runs"]["0/1"]==parent,"conditional branch changed a surviving attacker game")
            counts["skipped_after_survival"]+=1
            counts["avoided_replacements"]+=p["base"]!=p["alternate"]
        else:
            require(case==probe,"conditional branch changed the teacher after loss of the attacker")
            counts["eligible_after_removal"]+=1
            counts["adopted_after_removal"]+=p["base"]!=p["alternate"]
    return dict(counts)


def load(directory):
    roster = model_roster()
    weights = [w for name in ("candidate","historical") for w in roster[name]["weights"]]
    parent = json.loads(PARENT.read_bytes())
    payload = " ".join(map(str,(START,COUNT,*weights)))+" "+model_arguments(parent,"own_visits")
    expected = {"/".join(map(str,cell)) for cell in product(range(START,START+COUNT),range(4),(1,2),(0,1))}
    broad = parse(native(directory,"exchange-natural",payload),expected,False)
    gated = parse(native(directory,"exchange-natural-recovery",payload),expected,False)
    verify_parent(directory,broad,payload)
    return broad,gated


def compare(broad, gated):
    reporting={}
    for name, source in (("wide",broad),("recovery",gated)):
        desks={f"{p}/{t}":panel({k:v for k,v in source.items() if k.endswith(f"/{p}/{t}")})
               for p,t in product((1,2),(0,1))}
        all_games=panel(source)
        score=all_games["comparison"]["score_outer_interval"]
        loss_ok=all(d["comparison"]["extra_losses_by_140"]<=MAX_EXTRA_LOSS*d["baseline"]["n"]
                    for d in desks.values())
        reporting[name]=dict(all=all_games,desks=desks,loss_budget_passed=loss_ok,
                             qualified=score[0]>=MIN_GAIN and loss_ok)
    return reporting


def main(audit):
    directory = ROOT/"build/exchange-recovery"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    wide,recovery = load(directory)
    counts = check_gated(wide,recovery)
    comparisons = compare(wide,recovery)
    qualified = comparisons["recovery"]["qualified"]
    decision = "confirm_under_new_partner" if qualified else "diagnose_pre_elimination_progress"
    receipt = dict(counts=counts, comparisons=comparisons, decision=decision, wide=wide, recovery=recovery)
    persist(directory/"summary.json",encode(receipt),audit)
    print(json.dumps(dict(counts=counts, comparisons=comparisons, decision=decision),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
