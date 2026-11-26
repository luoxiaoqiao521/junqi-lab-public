"""Compare archived bomb replies with the actual historical model scores and play."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
import subprocess
import sys

from audit_replay_curriculum_roots import CONFIG, ROOT, RULES, compile_native, prepare, require
from audit_qq_archive import LIMIT, decode
from audit_human_timing import sanitized_tape
from compare_replay_first_followthrough import write_result
from pilot_relay_candidate_budget import ROOTS, setup


FIELDS = ("capture_ply","ally","status","delay","mobile_bombs","legal_replies","bomb_moved_elsewhere",
          "parent_mass_ppm","child_mass_ppm")
STATUS = ("target_gone","responder_out","no_turn_before_end","no_legal_reply","declined","taken")
SOURCES = ("archived","parent","trained")


def labeled_paths():
    for index, root in enumerate(ROOTS):
        for path in sorted(root.rglob("*.jgs"),key=lambda p:p.as_posix().casefold()):
            if any("双明" in part for part in path.relative_to(root).parts[:-1]):
                yield index,path


def labeled(feed, provenance):
    """Attach directory provenance without publishing filenames or opening sealed families."""
    family_key,*_ = setup(ROOT)
    sources = provenance["sources"]
    additions, counts = {},Counter()
    for origin,path in labeled_paths():
        counts["tagged_paths"]+=1
        try:
            require(path.stat().st_size<=LIMIT,"oversize replay")
            raw=path.read_bytes()
            key,layouts,_,events=decode(raw)
        except (OSError,ValueError):
            counts["rejected_jgs"]+=1
            continue
        family=family_key(layouts).hex()
        if int(family[:8],16)%4==0:
            counts["heldout_paths"]+=1
            continue
        tape=sanitized_tape(raw,events)
        digest=sha256(bytes.fromhex(tape)).hexdigest()
        if family in sources:
            first=sources[family]
            if first["tape"]==digest:
                counts["duplicate_family"]+=family in additions
                first["mode_provenance"]="directory_double_visible"
            else:
                counts["different_replay_same_family"]+=1
            continue
        additions[family]=tape
        sources[family]=dict(source=origin,replay=key[:16],events=len(events),tape=digest,
                             mode_provenance="directory_double_visible")
    for source in sources.values():
        source.setdefault("mode_provenance","unknown")
    require(counts["tagged_paths"]>0 and any(s["mode_provenance"]=="directory_double_visible"
            for s in sources.values()),"no directory-labeled double-visible replays reached the comparison")
    provenance["directory_inventory"]=dict(counts) | {"added":len(additions)}
    wire=feed+"".join(f"{family} {tape}\n" for family,tape in additions.items())
    provenance["feed_sha256"]=sha256(wire.encode("ascii")).hexdigest()
    return wire


def parse(stdout, source_ids):
    details,games,groups = {},{},defaultdict(dict)
    for line in stdout.splitlines():
        label,family,*raw = line.split()
        values = list(map(int,raw))
        require(family in source_ids,"unknown source family")
        if label=="D":
            require(family not in details and len(values)==20,"invalid replay integrity row")
            details[family] = values[:4]
            continue
        if label=="G":
            source,plies,result = values
            require((family,source) not in games and source in range(len(SOURCES))
                    and plies>0 and result in range(1,5),"invalid game endpoint")
            games[family,source]=dict(plies=plies,outcome=result)
            continue
        require(label=="B","unrecognized bomb reply line")
        source,*values = values
        row = dict(zip(FIELDS,values,strict=True))
        key = (row["capture_ply"],row["ally"])
        require(0<=source<len(SOURCES) and key not in groups[family,source],"duplicate bomb reply window")
        check(row)
        groups[family,source][key] = row
    require(details.keys()==source_ids,"incomplete game ledger")
    require(games.keys()=={(f,s) for f,d in details.items() if not any(d[2:]) for s in range(len(SOURCES))},
            "missing comparison game endpoint")
    for (family,source),items in groups.items():
        require(not any(details[family][2:]),"unverified source contributed to the comparison")
        require({(ply,1-ally) for ply,ally in items}==items.keys(),"capture missing a responder")
    return details, games, groups


def check(row):
    position = row["status"]
    require(row["capture_ply"]>0 and row["delay"]>=0 and row["ally"] in (0,1) and 0<=position<6,
            "invalid bomb reply record")
    require((row["legal_replies"]>0)==(position in (4,5)) and 0<=row["legal_replies"]<=row["mobile_bombs"]<=2,
            "invalid legal bomb count")
    require(all(0<=row[k]<=1000000 and (position>=4 or not row[k])
                for k in ("parent_mass_ppm","child_mass_ppm")), "invalid counterfactual policy probabilities")
    require(row["bomb_moved_elsewhere"] in (0,1) and (not row["bomb_moved_elsewhere"] or position in (3,4)),
            "invalid alternate bomb action")


def summarize(groups):
    panels = {}
    for source,name in enumerate(SOURCES):
        panel = {}
        for ally,role in enumerate(("self","ally")):
            blocks = [(family,row) for (family,s),items in groups.items() if s==source
                      for row in items.values() if row["ally"]==ally]
            offers = [(family,row) for family,row in blocks if row["legal_replies"]]
            panel[role] = dict(windows=len(blocks),statuses=dict(Counter(STATUS[r["status"]] for _,r in blocks)),
                opportunities=len(offers),opportunity_families=len({f for f,_ in offers}),
                used=sum(r["status"]==5 for _,r in offers),used_families=len({f for f,r in offers if r["status"]==5}),
                no_mobile_bomb=sum(r["status"]==3 and r["mobile_bombs"]==0 for _,r in blocks),
                bomb_moved_elsewhere=sum(r["bomb_moved_elsewhere"] for _,r in offers),
                expected_parent=sum(r["parent_mass_ppm"] for _,r in offers)/1e6,
                expected_trained=sum(r["child_mass_ppm"] for _,r in offers)/1e6)
        panels[name] = panel
    return panels


def main(audit):
    directory=ROOT/"build/replay-bomb-replies"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        compile_native(directory)
    feed,provenance=prepare()
    feed=labeled(feed,provenance)
    process=subprocess.run([directory/"takeover.exe",CONFIG,RULES,"960","bomb-replies"],
                           input=feed,encoding="utf8",capture_output=True,timeout=120)
    require(process.returncode==0,f"native comparison failed: {process.stderr.strip()}")
    source_ids=provenance["sources"].keys()
    details,games,groups=parse(process.stdout,source_ids)
    verified=sum(not any(value[2:]) for value in details.values())
    require(verified>0,"no verified replays")
    require(all(any(key[1]==s for key in groups) for s in range(len(SOURCES))),"missing comparison arm")
    panels=summarize(groups)
    labels={"all":panels}
    for key in ("directory_double_visible","unknown"):
        labels[key]=summarize({(f,s):g for (f,s),g in groups.items()
                              if provenance["sources"][f]["mode_provenance"]==key})
    write_result(directory,dict(verified=verified,provenance=provenance,integrity=details,panels=labels,
                 games=[dict(family=f,source=SOURCES[s],**r) for (f,s),r in games.items()],
                 windows=[dict(family=f,source=SOURCES[s],**r) for (f,s),g in groups.items() for r in g.values()]),audit)
    print(json.dumps({"verified":verified,"directory_inventory":provenance["directory_inventory"],
                      "panels":labels},sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
